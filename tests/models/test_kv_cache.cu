#include "models/qwen3.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
int main(int argc,char **argv) {
 try {
    if(argc<2) throw std::runtime_error("model directory required");
    qwe::Qwen3Options options; options.max_context=argc>3 ? std::stoi(argv[3]) : 64;
    if(argc>2 && std::string(argv[2])!="single") options.devices={0,1};
    if(argc>2 && std::string(argv[2])=="tp") options.tensor_parallel=true;
    qwe::Qwen3 plain(argv[1],options); options.kv_cache=true;
    qwe::Qwen3 cached(argv[1],options);
    auto compare=[&](const std::vector<int> &ids,bool full=false) {
        auto a=plain.forward(ids,!full),b=cached.forward(ids,!full);
        float max_error=0;
        for(size_t i=0;i<a.size();++i) {
            float error=std::fabs(a.data()[i]-b.data()[i]); max_error=std::max(max_error,error);
            if(!std::isfinite(b.data()[i]) || error>0.002f+0.0001f*std::fabs(a.data()[i]))
                throw std::runtime_error("cached logits mismatch");
        }
        std::cout<<"length="<<ids.size()<<" full="<<full<<" max_abs="<<max_error<<"\n";
    };
    std::vector<int> ids;
    for(int n:{1,2,4,8,16,32,64,257,512}) { if(n>options.max_context) continue; ids.resize(n,9707); compare(ids); }
    ids[0]=11;compare(ids); // changed prefix must prefill
    cached.reset_cache();compare(ids);
    ids.resize(4);compare(ids);compare(ids); // shorter and identical input
    compare(ids,true);ids.push_back(1879);compare(ids); // full-output invalidates cache
    bool rejected=false;try {cached.forward(std::vector<int>(options.max_context+1,9707),true);}catch(...){rejected=true;}
    if(!rejected) throw std::runtime_error("missing capacity rejection");
    cached.reset_cache();compare({9707,11});
    std::cout<<"KV cache numerical and invalidation checks passed bytes="<<cached.kv_cache_bytes()<<"\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
