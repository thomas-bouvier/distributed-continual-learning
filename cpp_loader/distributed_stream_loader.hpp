#ifndef __DISTRIBUTED_STREAM_LOADER
#define __DISTRIBUTED_STREAM_LOADER

#include <torch/extension.h>
#include <thallium.hpp>

#include "stream_loader_provider.hpp"

namespace tl = thallium;

class distributed_stream_loader_t {
    stream_loader_provider_t* provider;

public:
    distributed_stream_loader_t(unsigned int _K, unsigned int _N, unsigned int _C, int64_t seed,
        uint16_t id, std::vector<std::pair<std::string, int>> endpoints);
    ~distributed_stream_loader_t();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &labels,
        const torch::Tensor &aug_samples, const torch::Tensor &aug_labels, const torch::Tensor &aug_weights);
    int wait();
    size_t get_rehearsal_size();
    size_t get_history_count();
};

#endif
