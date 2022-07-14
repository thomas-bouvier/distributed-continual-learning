#include "distributed_stream_loader.hpp"

#include <unordered_set>
#include <assert.h>

#define __DEBUG
#include "debug.hpp"

distributed_stream_loader_t::distributed_stream_loader_t(unsigned int _K, unsigned int _N, unsigned int _C,
    int64_t seed, uint16_t id, std::vector<std::pair<std::string, int>> endpoints) {\
    tl::engine myServer("tcp://127.0.0.1:1235", THALLIUM_SERVER_MODE);
    std::cout << "Server running at address " << myServer.self()
        << " with provider id " << id << std::endl;
    
    provider = new stream_loader_provider_t(myServer, id, _K, _N, _C, seed, endpoints);
}

void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &labels,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_labels, const torch::Tensor &aug_weights) {
    provider->accumulate(samples, labels, aug_samples, aug_labels, aug_weights);
}

int distributed_stream_loader_t::wait() {
    return provider->wait();
}

size_t distributed_stream_loader_t::get_rehearsal_size() {
    return provider->get_rehearsal_size();
}

size_t distributed_stream_loader_t::get_history_count() {
    return provider->get_history_count();
}

distributed_stream_loader_t::~distributed_stream_loader_t() {
    delete provider;
}
