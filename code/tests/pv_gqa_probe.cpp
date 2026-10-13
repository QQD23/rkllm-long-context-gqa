#include "pv_gqa_engine.h"
#include <thread>
#include <cmath>
int main(int argc,char**argv){
 const int K=argc>1?atoi(argv[1]):8256,M=64,N=256,H=16,VH=4,W=3;
 if(K<32||K>16384)return 2;
 // Deliberate row/head padding checks that the kernel obeys tensor strides.
 int vk=K+32,pk=K+16,dn=N+8;
 std::vector<__fp16> v(size_t(VH)*N*vk);
 std::vector<float> p(size_t(H)*M*pk),d(size_t(H)*M*dn,-123.f);
 double sums[4]={};
 for(int k=0;k<K;++k){double u=k%5+1,z=k%3,b=k%7-3,c=k%11-5;sums[0]+=u*b;sums[1]+=u*c;sums[2]+=z*b;sums[3]+=z*c;}
 for(int h=0;h<H;++h)for(int m=0;m<M;++m)for(int k=0;k<K;++k)p[(h*M+m)*pk+k]=float((m%7+1)*(k%5+1))/1024+float((h%5+1)*(k%3))/2048;
 for(int vh=0;vh<VH;++vh)for(int n=0;n<N;++n)for(int k=0;k<K;++k)v[(vh*N+n)*vk+k]=float((n%9+1)*(k%7-3))/16+float((vh+1)*(k%11-5))/16;
 pv_gqa::View vv{(char*)v.data(),{2,uint64_t(vk*2),uint64_t(N*vk*2),0}},pp{(char*)p.data(),{4,uint64_t(pk*4),uint64_t(M*pk*4),0}},dd{(char*)d.data(),{4,uint64_t(dn*4),uint64_t(M*dn*4),0}};
 pv_gqa::Engine engine[W];
 for(int rep=0;rep<3;++rep){
  pv_gqa::Stats s[W];double worker_ms[W];std::thread t[W];auto begin=pv_gqa::Clock::now();
  for(int tid=0;tid<W;++tid)t[tid]=std::thread([&,tid]{auto b=pv_gqa::Clock::now();s[tid]=engine[tid].compute(vv,pp,dd,K,M,N,H,VH,tid,W);worker_ms[tid]=pv_gqa::elapsed(b);});
  for(auto& th:t)th.join();double wall=pv_gqa::elapsed(begin),err=0;bool padding=true;
  for(int h=0;h<H;++h)for(int m=0;m<M;++m)for(int n=0;n<dn;++n){
   if(n>=N){padding&=d[(h*M+m)*dn+n]==-123.f;continue;}
   double a=double(m%7+1)/1024,b=double(h%5+1)/2048,c=double(n%9+1)/16,e=double(h/4+1)/16;
   double expected=a*c*sums[0]+a*e*sums[1]+b*c*sums[2]+b*e*sums[3];
   double got=d[(h*M+m)*dn+n];if(!std::isfinite(got))return 4;err=std::max(err,std::abs(got-expected));
  }
  int calls=0;size_t bytes=0;for(auto& st:s){calls+=st.calls;bytes+=st.b_bytes;}
  printf("{\"K\":%d,\"rep\":%d,\"warmup\":%s,\"wall_ms\":%.6f,\"tid0_ms\":%.6f,\"tid0_setup_ms\":%.6f,\"tid0_pack_a_ms\":%.6f,\"tid0_pack_b_ms\":%.6f,\"tid0_bind_ms\":%.6f,\"tid0_run_ms\":%.6f,\"tid0_sum_ms\":%.6f,\"calls\":%d,\"v_pack_bytes\":%zu,\"maxabs\":%.9g,\"padding_ok\":%s}\n",K,rep,rep==0?"true":"false",wall,worker_ms[0],s[0].setup,s[0].pack_a,s[0].pack_b,s[0].bind,s[0].run,s[0].sum,calls,bytes,err,padding?"true":"false");fflush(stdout);
  if(err>.002||!padding)return 3;
 }
}
