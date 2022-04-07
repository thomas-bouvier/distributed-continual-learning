#ifndef __STREAM_LOADER
#define __STREAM_LOADER

#include <torch/extension.h>
#include <unordered_map>
#include <iostream>
#include <tuple>
#include <random>

typedef std::tuple<std::string, torch::Tensor> sample_t;
typedef std::vector<sample_t> sample_list_t;
typedef std::vector<torch::Tensor> buffer_t;
typedef std::unordered_map<std::string, buffer_t> rehearsal_map_t;

class stream_loader_t {
    unsigned int K, N;
    std::default_random_engine rand_gen;
    std::uniform_int_distribution<unsigned int> rand_class, rand_sample;
    rehearsal_map_t rehearsal_map;

public:
    stream_loader_t(unsigned int _K, unsigned int _N, int64_t seed);
    void accumulate(const sample_list_t &samples);
    sample_list_t get_samples(unsigned int R);
};

#endif
