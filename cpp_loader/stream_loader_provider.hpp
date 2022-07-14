#ifndef __STREAM_LOADER_PROVIDER
#define __STREAM_LOADER_PROVIDER

#include <torch/extension.h>
#include <thallium.hpp>
#include <unordered_map>
#include <iostream>
#include <random>

namespace tl = thallium;

typedef std::vector<torch::Tensor> buffer_t;
typedef std::unordered_map<int, std::pair<double, buffer_t>> rehearsal_map_t;
typedef std::unordered_map<int, int> rehearsal_counts_t;

class stream_loader_provider_t : public tl::provider<stream_loader_provider_t> {
    const size_t MAX_QUEUE_SIZE = 1024;

    unsigned int K, N, C;
    std::default_random_engine rand_gen;
    rehearsal_map_t rehearsal_map;
    rehearsal_counts_t counts;
    size_t history_count = 0;
    size_t rehearsal_size = 0;

    struct queue_item_t {
        int aug_size = 0;
        torch::Tensor samples, labels, aug_samples, aug_labels, aug_weights;
        queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_labels,
                const torch::Tensor &_aug_samples, const torch::Tensor &_aug_labels, const torch::Tensor &_aug_weights) :
            samples(_samples), labels(_labels), aug_samples(_aug_samples), aug_labels(_aug_labels), aug_weights(_aug_weights) { }
        queue_item_t() { }
    };
    std::deque<queue_item_t> request_queue, response_queue;
    std::mutex request_mutex;
    std::condition_variable request_cond;
    std::thread async_thread;

    rehearsal_map_t selected_samples;
    std::vector<tl::endpoint> provider_handles;
    tl::remote_procedure get_samples_procedure;
    void get_samples(unsigned int index);
    void get_remote_samples(const tl::request& req, unsigned int index);

    void async_process();

public:
    stream_loader_provider_t(tl::engine& e, uint16_t provider_id,
        unsigned int _K, unsigned int _N, unsigned int _C, int64_t seed,
        const std::vector<std::pair<std::string, int>>& endpoints);
    ~stream_loader_provider_t();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &labels,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_labels, const torch::Tensor &aug_weights);
    int wait();
    size_t get_rehearsal_size();
    size_t get_history_count();
};

#endif
