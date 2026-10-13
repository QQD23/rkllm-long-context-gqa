// RK3588 experimental GQA row grouping. No KV truncation or clock writes.
#pragma once
#include "rknn_matmul_api.h"
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
namespace pv_gqa {
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
inline void check(int rc,const char* what){if(rc){fprintf(stderr,"PV_GQA_FATAL %s rc=%d\n",what,rc);std::abort();}}
struct View {char* data; uint64_t stride[4];};
struct Stats {double setup=0,pack_a=0,pack_b=0,bind=0,run=0,sum=0;int calls=0;size_t b_bytes=0;};
struct Op {
 rknn_matmul_ctx ctx=0; rknn_matmul_io_attr io{};
 rknn_tensor_mem *a=nullptr,*b=nullptr,*c=nullptr;
 Op(int rows,int k,int n,int core){
  rknn_matmul_info i{};i.M=rows;i.K=k;i.N=n;i.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;i.B_layout=1;i.iommu_domain_id=2;
  check(rknn_matmul_create(&ctx,&i,&io),"create");check(rknn_matmul_set_core_mask(ctx,(rknn_core_mask)(1<<core)),"core");
  if(io.B.n_dims!=4||io.B.dims[0]!=unsigned(n/16)||io.B.dims[1]!=unsigned(k/32)||io.B.dims[2]!=16||io.B.dims[3]!=32||io.A.size<size_t(rows*k*2)||io.C.size<size_t(rows*n*4)){fprintf(stderr,"PV_GQA unsupported layout\n");abort();}
  a=rknn_create_mem(ctx,io.A.size);b=rknn_create_mem(ctx,io.B.size);c=rknn_create_mem(ctx,io.C.size);
  if(!a||!b||!c)abort();check(rknn_matmul_set_io_mem(ctx,c,&io.C),"bind_c");
 }
 ~Op(){if(a)rknn_destroy_mem(ctx,a);if(b)rknn_destroy_mem(ctx,b);if(c)rknn_destroy_mem(ctx,c);if(ctx)rknn_matmul_destroy(ctx);}
};
struct Engine {
 std::map<std::tuple<int,int,int,int>,std::unique_ptr<Op>> ops;
 std::vector<float> accum;
 // Each worker owns whole KV heads, matching the original RKLLM worker partition.
 Stats compute(View v,View p,View dst,int K,int M,int N,int H,int VH,int tid,int workers,int block=4096){
  if(K<=0||K>16384||M<=0||M>256||N!=256||VH<=0||H%VH||H/VH>16||tid<0||tid>=workers||workers<1||workers>3||block!=4096)abort();
  Stats s;const int G=H/VH,rows=G*M;accum.resize(size_t(rows)*N);
  for(int vh=tid;vh<VH;vh+=workers){
   std::fill(accum.begin(),accum.end(),0.f);
   for(int off=0;off<K;off+=block){
    const int used=std::min(block,K-off),bk=(used+31)&~31;
    auto t=Clock::now();auto key=std::make_tuple(rows,bk,N,tid);auto it=ops.find(key);
    if(it==ops.end())it=ops.emplace(key,std::make_unique<Op>(rows,bk,N,tid)).first;
    Op& o=*it->second;s.setup+=elapsed(t);t=Clock::now();
    // One B packing per KV head/block; all G query heads reuse it in one MatMul.
    for(int nb=0;nb<N/16;++nb)for(int kb=0;kb<bk/32;++kb)for(int ni=0;ni<16;++ni){
     int count=std::max(0,std::min(32,used-kb*32));
     char* src=v.data+vh*v.stride[2]+(nb*16+ni)*v.stride[1]+size_t(off+kb*32)*v.stride[0];
     auto* d=(__fp16*)o.b->virt_addr+((nb*(bk/32)+kb)*16+ni)*32;
     if(v.stride[0]==2)memcpy(d,src,count*2);else for(int k=0;k<count;++k)memcpy(d+k,src+k*v.stride[0],2);
     if(count<32)memset(d+count,0,(32-count)*2);
    }
    s.b_bytes+=size_t(used)*N*2;s.pack_b+=elapsed(t);t=Clock::now();
    for(int g=0;g<G;++g)for(int m=0;m<M;++m){
     char* src=p.data+(vh*G+g)*p.stride[2]+m*p.stride[1]+size_t(off)*p.stride[0];
     auto* d=(__fp16*)o.a->virt_addr+(g*M+m)*bk;int k=0;
     if(p.stride[0]==4)for(;k+4<=used;k+=4)vst1_f16(d+k,vcvt_f16_f32(vld1q_f32((float*)(src+k*4))));
     for(;k<used;++k){float x;memcpy(&x,src+k*p.stride[0],4);d[k]=x;}
     if(used<bk)memset(d+used,0,(bk-used)*2);
    }
    s.pack_a+=elapsed(t);t=Clock::now();
    check(rknn_matmul_set_io_mem(o.ctx,o.a,&o.io.A),"bind_a");check(rknn_matmul_set_io_mem(o.ctx,o.b,&o.io.B),"bind_b");
    s.bind+=elapsed(t);t=Clock::now();check(rknn_matmul_run(o.ctx),"run");s.run+=elapsed(t);++s.calls;t=Clock::now();
    const float* c=(float*)o.c->virt_addr;
    for(size_t j=0;j<accum.size();j+=4)vst1q_f32(accum.data()+j,vaddq_f32(vld1q_f32(accum.data()+j),vld1q_f32(c+j)));
    s.sum+=elapsed(t);
   }
   auto t=Clock::now();
   for(int g=0;g<G;++g)for(int m=0;m<M;++m){
    char* d=dst.data+(vh*G+g)*dst.stride[2]+m*dst.stride[1];const float* a=accum.data()+(g*M+m)*N;
    if(dst.stride[0]==4)memcpy(d,a,N*4);else for(int n=0;n<N;++n)memcpy(d+n*dst.stride[0],a+n,4);
   }
   s.sum+=elapsed(t);
  }
  return s;
 }
};
}
