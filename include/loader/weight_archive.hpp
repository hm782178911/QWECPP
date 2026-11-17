#pragma once
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include "cJSON.h"
extern "C" {
#include "loader/safetensors.h"
}
namespace qwe {
// Owns lazily mapped checkpoint shards. Returned pointers live with this archive.
class WeightArchive {
    struct File {
        SafeTensorFile value{};
        explicit File(const std::string &path) {
            if(safetensors_open(&value,path.c_str())) throw std::runtime_error(value.error);
        }
        ~File() { safetensors_close(&value); }
    };
    std::string directory;
    std::map<std::string,std::string> index;
    std::map<std::string,std::unique_ptr<File>> files;
    bool indexed=false;
public:
    explicit WeightArchive(const std::string &dir): directory(dir) {
        std::ifstream input(dir+"/model.safetensors.index.json");
        if(!input) return;
        indexed=true;
        std::string text((std::istreambuf_iterator<char>(input)),{});
        std::unique_ptr<cJSON,decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()),cJSON_Delete);
        auto *weights=cJSON_GetObjectItemCaseSensitive(root.get(),"weight_map");
        if(!cJSON_IsObject(weights) || !weights->child) throw std::runtime_error("invalid checkpoint weight_map");
        for(auto *entry=weights->child;entry;entry=entry->next) {
            if(!entry->string || !cJSON_IsString(entry)) throw std::runtime_error("invalid checkpoint index entry");
            std::string name=entry->valuestring;
            if(name.empty() || name.find('/')!=std::string::npos || name.find('\\')!=std::string::npos ||
               name.find("..")!=std::string::npos || name.find(':')!=std::string::npos ||
               name.size()<12 || name.substr(name.size()-12)!=".safetensors")
                throw std::runtime_error("invalid checkpoint shard filename");
            if(!index.emplace(entry->string,name).second) throw std::runtime_error("duplicate checkpoint tensor index");
        }
    }
    const SafeTensorFile &file_for(const std::string &tensor) {
        std::string name="model.safetensors";
        if(indexed) {
            auto it=index.find(tensor);
            if(it==index.end()) throw std::runtime_error("missing tensor in checkpoint index: "+tensor);
            name=it->second;
        }
        auto it=files.find(name);
        if(it==files.end()) it=files.emplace(name,std::make_unique<File>(directory+"/"+name)).first;
        return it->second->value;
    }
};
}
