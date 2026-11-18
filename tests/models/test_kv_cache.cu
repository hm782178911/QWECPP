#include "models/qwen3.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
int main(int argc,char **argv) {
 try {
    if(argc<2) throw std::runtime_error("model directory required");
    if (argc > 2 && (std::string(argv[2]) == "pd" || std::string(argv[2]) == "pd-host" ||
                     std::string(argv[2]) == "pd-pp" || std::string(argv[2]) == "pd-tp")) {
        qwe::Qwen3Options options;
        options.max_context = 64;
        options.kv_cache = true;
        std::string mode = argv[2];
        bool grouped = mode == "pd-pp" || mode == "pd-tp";
        options.tensor_parallel = mode == "pd-tp";
        options.devices = grouped ? std::vector<int>{0, 1} : std::vector<int>{0};
        qwe::Qwen3 baseline(argv[1], options);
        options.devices = grouped ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{0, 1};
        options.prefill_decode = true;
        options.profile = true;
        options.force_host_transfer = std::string(argv[2]) == "pd-host";
        qwe::Qwen3 pd(argv[1], options);
        auto compare = [&](const std::vector<int> &ids, bool handoff) {
            auto a = baseline.forward(ids, true), b = pd.forward(ids, true);
            if (a.size() != b.size() ||
                std::max_element(a.begin(), a.end()) - a.begin() !=
                std::max_element(b.begin(), b.end()) - b.begin())
                throw std::runtime_error("PD logits shape or argmax mismatch");
            double mean = 0;
            float max_error = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                float error = std::fabs(a.data()[i] - b.data()[i]);
                if (!std::isfinite(error) || error > 0.002f + 0.0001f * std::fabs(a.data()[i]))
                    throw std::runtime_error("PD logits mismatch");
                max_error = std::max(max_error, error);
                mean += error;
            }
            auto stats = pd.last_kv_transfer_stats();
            if ((stats.calls > 0) != handoff ||
                (options.force_host_transfer && stats.host_calls != stats.calls))
                throw std::runtime_error("unexpected PD handoff");
            std::cout << "length=" << ids.size() << " max_abs=" << max_error
                      << " mean_abs=" << mean / a.size() << " bytes=" << stats.bytes
                      << " p2p=" << stats.p2p_calls << " host=" << stats.host_calls << "\n";
        };
        compare({9707, 11}, false);
        compare({9707, 11, 1879}, true);
        compare({9707, 11, 1879, 42}, false);
        compare({11, 9707}, false);
        compare({11, 9707, 42, 43}, true);
        pd.reset_cache();
        baseline.reset_cache();
        compare({42}, false);
        compare({42, 11}, true);
        return 0;
    }
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
