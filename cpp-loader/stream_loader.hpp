#ifndef __STREAM_LOADER
#define __STREAM_LOADER

#include <torch/extension.h>
#include <map>
#include <iostream>
#include <tuple>

typedef std::tuple<std::string, torch::Tensor> sample_t;
typedef std::vector<sample_t> sample_list_t;
typedef std::vector<torch::Tensor> buffer_t;
typedef std::map<std::string, buffer_t> rehearsal_map_t;

class stream_loader_t {
    int K, N;
public:
    stream_loader_t(int _K, int _N) : K(_K), N(_N) { }
    void accumulate(const sample_list_t &samples);    
};

#endif
