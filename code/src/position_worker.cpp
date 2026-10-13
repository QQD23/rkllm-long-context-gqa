// RKLLM v1.3.0 adapter. These are OUTPUT callback boundaries, not operator boundaries.
#include <cstddef>
#include <cstdint>
#include <rkllm.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#include <time.h>
#include <sys/prctl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstdlib>
#endif

// Optional experiment gate: odd generation means inside rkllm_run. No polling IO.
struct InferenceGate {
#ifndef _WIN32
    uint64_t* data=nullptr; uint64_t generation=0;
    InferenceGate() {
        const char* path=std::getenv("WP_INFERENCE_GATE"); if(!path)return;
        int fd=open(path,O_RDWR|O_CREAT|O_TRUNC|O_CLOEXEC,0600);
        if(fd<0)throw std::runtime_error("gate open");
        if(ftruncate(fd,4096)){close(fd);throw std::runtime_error("gate resize");}
        void* p=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
        if(p==MAP_FAILED)throw std::runtime_error("gate mmap");
        data=static_cast<uint64_t*>(p);__atomic_store_n(data,generation,__ATOMIC_RELEASE);
    }
    void enter(){if(data)__atomic_store_n(data,++generation,__ATOMIC_RELEASE);}
    void leave(){if(data)__atomic_store_n(data,++generation,__ATOMIC_RELEASE);}
    ~InferenceGate(){if(data){if(generation&1)leave();munmap(data,4096);}}
#else
    void enter(){} void leave(){}
#endif
};

static int64_t now_ns() {
#ifdef _WIN32
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
#endif
}
struct Event { int state, token; bool has_result; int64_t ns, cost_ns; };
struct State {
    std::mutex mutex;
    bool trace=false, overflow=false, error=false;
    size_t used=0;
    int finishes=0;
    int generated=0,progress_every=0;
    uint64_t text_hash=14695981039346656037ULL;
    size_t text_bytes=0;
    RKLLMPerfStat perf{};
    std::vector<Event> events;
};
static int callback(RKLLMResult* r, void* userdata, LLMCallState state) {
    auto& s=*static_cast<State*>(userdata);
    int64_t entry=s.trace?now_ns():0;
    std::lock_guard<std::mutex> lock(s.mutex);
    if(state==RKLLM_RUN_FINISH) { ++s.finishes; if(r)s.perf=r->perf; }
    if(state==RKLLM_RUN_ERROR)s.error=true;
    if((state==RKLLM_RUN_NORMAL||state==RKLLM_RUN_WAITING)&&r&&r->token_id>=0){
        ++s.generated;
        if(s.generated==1)
            std::cout<<"\n@@WP@@{\"event\":\"first_token\",\"tokens\":1,\"ns\":"<<entry<<"}\n"<<std::flush;
        if(s.progress_every>0&&s.generated%s.progress_every==0)
            std::cout<<"\n@@WP@@{\"event\":\"progress\",\"tokens\":"<<s.generated<<"}\n"<<std::flush;
    }
    if((state==RKLLM_RUN_NORMAL||state==RKLLM_RUN_WAITING)&&r&&r->text) {
        for(const unsigned char* p=reinterpret_cast<const unsigned char*>(r->text);*p;++p) {
            s.text_hash^=*p; s.text_hash*=1099511628211ULL; ++s.text_bytes;
        }
    }
    if(s.used==s.events.size()) { s.overflow=true; return 0; }
    // Preserve every state and repeated token ID. WAITING is UTF-8 buffering.
    auto& e=s.events[s.used++];
    e={int(state),r&&(state==RKLLM_RUN_NORMAL||state==RKLLM_RUN_WAITING)?r->token_id:-1,r!=nullptr,entry,0};
    if(s.trace)e.cost_ns=now_ns()-entry;
    return 0;
}
static void number(std::ostream& o, double v) { if(std::isfinite(v))o<<v;else o<<"null"; }
int main(int argc,char** argv) {
    if(argc!=10) { std::cerr<<"MODEL INPUT OUTPUT_DIR MAX_NEW CONTEXT CPU_MASK EMBED_FLASH prompt|tokens SDK_ID\n";return 2; }
#ifndef _WIN32
    auto parent=getppid(); prctl(PR_SET_PDEATHSIG,SIGTERM); if(getppid()!=parent)return 2;
#endif
    try {
        InferenceGate gate;
        int max_new=std::stoi(argv[4]), context=std::stoi(argv[5]);
        unsigned mask=std::stoul(argv[6]);
        if(max_new<2||max_new>65536||context<=max_new||mask>255)throw std::runtime_error("invalid settings");
        std::ifstream input_file(argv[2]); if(!input_file)throw std::runtime_error("cannot read input");
        std::string prompt((std::istreambuf_iterator<char>(input_file)),{});
        if(prompt.empty())throw std::runtime_error("empty input");
        bool token_input=std::string(argv[8])=="tokens";
        if(!token_input&&std::string(argv[8])!="prompt")throw std::runtime_error("bad input type");
        std::vector<int32_t> ids;
        if(token_input) {
            std::istringstream stream(prompt);int32_t id;
            while(stream>>id){if(id<0)throw std::runtime_error("negative input token");ids.push_back(id);}
            if(!stream.eof()||ids.empty()||ids.size()+max_new>size_t(context))throw std::runtime_error("bad input tokens/context");
        }
        RKLLMParam p=rkllm_createDefaultParam(); p.model_path=argv[1];
        p.max_new_tokens=max_new;p.max_context_len=context;p.top_k=1;p.top_p=1;p.temperature=0;
        p.repeat_penalty=1;p.frequency_penalty=0;p.presence_penalty=0;p.mirostat=0;
        p.is_async=false;p.ignore_eos_token=false;
#ifndef _WIN32
        // Explicit fixed-output performance experiment; default remains normal EOS.
        if(const char* flag=std::getenv("WP_IGNORE_EOS"))p.ignore_eos_token=std::string(flag)=="1";
#endif
        p.extend_param.base_domain_id=1;p.extend_param.embed_flash=std::stoi(argv[7]);
        if(mask){p.extend_param.enabled_cpus_mask=mask;p.extend_param.enabled_cpus_num=__builtin_popcount(mask);}
        RKLLMCallback cb{};cb.result_callback=callback;LLMHandle handle=nullptr;
        int rc=rkllm_init(&handle,&p,&cb);if(rc||!handle)return 3;
        std::cout<<"\n@@WP@@{\"event\":\"ready\"}\n"<<std::flush;
        std::string command;
#ifndef _WIN32
        auto boundary_begin=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"wp_boundary_begin"));
        auto boundary_end=reinterpret_cast<int(*)(const char*)>(dlsym(RTLD_DEFAULT,"wp_boundary_end"));
#endif
        while(std::getline(std::cin,command)&&command!="quit") {
            std::istringstream cmd(command);std::string label,mode,extra;cmd>>label>>mode;
            if(label.empty()||label.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos
                ||(mode!="baseline"&&mode!="events"&&mode!="activity")) { rc=2;break; }
            int output_limit=max_new;
            if(cmd>>extra){output_limit=std::stoi(extra);if(output_limit<2||output_limit>max_new||(cmd>>extra)){rc=2;break;}}
            if(rkllm_clear_kv_cache(handle,0,nullptr,nullptr)!=0){rc=4;break;}
            RKLLMInput in{};in.role="user";in.enable_thinking=false;
            if(token_input){in.input_type=RKLLM_INPUT_TOKEN;in.token_input.input_ids=ids.data();in.token_input.n_tokens=ids.size();}
            else {in.input_type=RKLLM_INPUT_PROMPT;in.prompt_input=prompt.c_str();}
            RKLLMInferParam infer{};infer.mode=RKLLM_INFER_GENERATE;infer.keep_history=0;
            infer.max_new_tokens=output_limit;
            State s;s.trace=mode!="baseline";s.events.resize(size_t(max_new)*4+1024);
#ifndef _WIN32
            if(const char* every=std::getenv("WP_PROGRESS_EVERY"))s.progress_every=std::max(0,std::atoi(every));
#endif
#ifndef _WIN32
            bool boundary_enabled=boundary_begin&&boundary_end&&s.trace;
            if(boundary_enabled&&boundary_begin()!=0){rc=6;break;}
#endif
            int64_t begin=now_ns();gate.enter();rc=rkllm_run(handle,&in,&infer,&s);gate.leave();int64_t end=now_ns();
            int boundary_rc=0;
#ifndef _WIN32
            if(boundary_enabled)boundary_rc=boundary_end((std::string(argv[3])+"/"+label+"_boundaries.csv").c_str());
#endif
            std::lock_guard<std::mutex> lock(s.mutex);
            bool valid=rc==0&&!s.error&&!s.overflow&&s.finishes==1&&boundary_rc==0;
            std::ofstream out(std::string(argv[3])+"/"+label+".json");
            out<<std::setprecision(15)<<"{\"schema_version\":1,\"clock\":\"CLOCK_MONOTONIC\",\"mode\":\""<<mode
               <<"\",\"start_ns\":"<<begin<<",\"end_ns\":"<<end<<",\"rc\":"<<rc
               <<",\"valid\":"<<(valid?"true":"false")<<",\"overflow\":"<<(s.overflow?"true":"false")
               <<",\"finish_count\":"<<s.finishes<<",\"text_hash\":\""<<s.text_hash<<"\",\"text_bytes\":"<<s.text_bytes
               <<",\"boundary_trace_rc\":"<<boundary_rc
               <<",\"perf\":{\"prefill_tokens\":"<<s.perf.prefill_tokens<<",\"generate_tokens\":"<<s.perf.generate_tokens
               <<",\"prefill_ms\":";number(out,s.perf.prefill_time_ms);out<<",\"generate_ms\":";number(out,s.perf.generate_time_ms);
            out<<"},\"callbacks\":[";
            for(size_t i=0;i<s.used;++i){const auto& e=s.events[i];if(i)out<<',';
                out<<"{\"state\":"<<e.state<<",\"token_id\":"<<e.token<<",\"has_result\":"<<(e.has_result?"true":"false")
                   <<",\"ns\":"<<e.ns<<",\"handler_ns\":"<<e.cost_ns<<'}';}
            out<<"]}\n";out.flush();valid=valid&&out.good();out.close();valid=valid&&!out.fail();
            std::cout<<"\n@@WP@@{\"event\":\"done\",\"label\":\""<<label<<"\",\"valid\":"<<(valid?"true":"false")<<"}\n"<<std::flush;
            if(!valid){rc=5;break;}
        }
        rkllm_destroy(handle);return rc;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
}
