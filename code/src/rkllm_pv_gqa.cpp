// Private adapter for the hash-pinned RKLLM 1.3.0 build; original disk library is untouched.
#include "pv_gqa_engine.h"
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
struct Tensor {int32_t type;char pad[12];int64_t ne[4];uint64_t nb[4];};
static char* data(Tensor*t){return *reinterpret_cast<char**>((char*)t+1776);}
static const char* name(Tensor*t){return (char*)t+1784;}
using Fn=void(*)(Tensor*,Tensor*,Tensor*,int,void*);
static Fn original=nullptr;static bool shadow;
static unsigned long long ns(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return (unsigned long long)t.tv_sec*1000000000ull+t.tv_nsec;}
static pv_gqa::View view(Tensor* t){pv_gqa::View v{};v.data=data(t);memcpy(v.stride,t->nb,sizeof(v.stride));return v;}
static void replacement(Tensor*v,Tensor*p,Tensor*d,int tid,void*ctx){
 bool match=v->type==1&&p->type==0&&d->type==0&&strstr(name(p),"kq_soft_max")&&v->ne[0]==p->ne[0]&&p->ne[0]>8192&&p->ne[1]>1;
 if(!match){original(v,p,d,tid,ctx);return;}
 int64_t workers=ctx?*(int64_t*)((char*)ctx+8):0;
 if(tid<0||tid>=workers)return;
 int K=p->ne[0],M=p->ne[1],N=v->ne[1],H=p->ne[2],VH=v->ne[2];
 if(workers<1||workers>3||M!=64||N!=256||H!=16||VH!=4||K>16384||p->ne[3]!=1||v->ne[3]!=1||d->ne[0]!=N||d->ne[1]!=M||d->ne[2]!=H||d->ne[3]!=1){fprintf(stderr,"PV_GQA unsupported tensor shape\n");abort();}
 static thread_local pv_gqa::Engine engine;
 static thread_local bool verified=false;
 static thread_local std::vector<float> candidate;
 candidate.resize(size_t(H)*M*N);
 pv_gqa::View out{(char*)candidate.data(),{4,uint64_t(N*4),uint64_t(M*N*4),uint64_t(H*M*N*4)}};
 auto start=ns();auto begin=pv_gqa::Clock::now();auto st=engine.compute(view(v),view(p),out,K,M,N,H,VH,tid,workers);double total=pv_gqa::elapsed(begin),originalms=0;
 if(shadow||!verified){
  begin=pv_gqa::Clock::now();original(v,p,d,tid,ctx);originalms=pv_gqa::elapsed(begin);
  double maxerr=0,ss=0,rr=0;bool finite=true;
  for(int h=0;h<H;++h)if((h/(H/VH))%workers==tid)for(int m=0;m<M;++m)for(int n=0;n<N;++n){
   float ref;memcpy(&ref,data(d)+h*d->nb[2]+m*d->nb[1]+n*d->nb[0],4);float got=candidate[(h*M+m)*N+n];
   finite&=std::isfinite(ref)&&std::isfinite(got);double diff=double(got)-ref;maxerr=std::max(maxerr,std::abs(diff));ss+=diff*diff;rr+=double(ref)*ref;
  }
  double rel=std::sqrt(ss/std::max(rr,1e-30));
  fprintf(stderr,"RKDUANYADIAN_VERIFY tid=%d K=%d M=%d N=%d H=%d VH=%d maxabs=%.9g relrms=%.9g tile_ms=%.3f original_ms=%.3f finite=%d backend=gqa\n",tid,K,M,N,H,VH,maxerr,rel,total,originalms,int(finite));
  if(!finite||!std::isfinite(rel)||rel>.01||maxerr>.02){fprintf(stderr,"PV_GQA numerical verification failed\n");abort();}verified=true;
 }
 if(!shadow)for(int h=0;h<H;++h)if((h/(H/VH))%workers==tid)for(int m=0;m<M;++m){
  char* dest=data(d)+h*d->nb[2]+m*d->nb[1];const float* src=candidate.data()+(h*M+m)*N;
  if(d->nb[0]==4)memcpy(dest,src,N*4);else for(int n=0;n<N;++n)memcpy(dest+n*d->nb[0],src+n,4);
 }
 fprintf(stderr,"PV_GQA_TIMING tid=%d K=%d total_ms=%.3f setup_ms=%.3f pack_a_ms=%.3f pack_b_ms=%.3f bind_ms=%.3f run_ms=%.3f sum_ms=%.3f calls=%d v_bytes=%zu\n",tid,K,total,st.setup,st.pack_a,st.pack_b,st.bind,st.run,st.sum,st.calls,st.b_bytes);
 if(tid==0)fprintf(stderr,"RKDUANYADIAN_PV K=%d M=%d N=%d H=%d block=4096 tile_ms=%.3f shadow=%d start_ns=%llu end_ns=%llu original_check_ms=%.3f backend=gqa\n",K,M,N,H,total,int(shadow),start,ns(),originalms);
}
static void jump(void*dst,void*target){uint32_t words[2]={0x58000050,0xd61f0200};memcpy(dst,words,8);memcpy((char*)dst+8,&target,8);}
__attribute__((constructor))static void install(){
 const char* mode=getenv("RKDUANYADIAN_MODE");if(!mode)return;
 shadow=strcmp(mode,"shadow")==0;if(!shadow&&strcmp(mode,"tile")){fprintf(stderr,"bad RKDUANYADIAN_MODE\n");abort();}
 Dl_info info{};if(!dladdr(dlsym(RTLD_DEFAULT,"rkllm_init"),&info))abort();
 char*entry=(char*)info.dli_fbase+0x318dd0;uint32_t expected[4]={0xd110c3ff,0xa9007bfd,0x910003fd,0xa90573fb};if(memcmp(entry,expected,16)){fprintf(stderr,"RKLLM opcode guard failed\n");abort();}
 void*tramp=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(tramp==MAP_FAILED)abort();memcpy(tramp,entry,16);jump((char*)tramp+16,entry+16);
 __builtin___clear_cache((char*)tramp,(char*)tramp+32);if(mprotect(tramp,4096,PROT_READ|PROT_EXEC))abort();original=(Fn)tramp;
 uintptr_t page=(uintptr_t)entry&~uintptr_t(4095);if(mprotect((void*)page,4096,PROT_READ|PROT_WRITE|PROT_EXEC))abort();jump(entry,(void*)&replacement);__builtin___clear_cache(entry,entry+16);if(mprotect((void*)page,4096,PROT_READ|PROT_EXEC))abort();
 fprintf(stderr,"RKDUANYADIAN installed mode=%s offset=0x318dd0 backend=gqa block=4096\n",mode);
}
