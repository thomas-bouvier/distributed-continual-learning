#include "stream_loader.hpp"

#include <unordered_set>
#include <assert.h>

#define __DEBUG
#include "debug.hpp"

using namespace torch::indexing;

stream_loader_t::stream_loader_t(unsigned int _K, unsigned int _N, unsigned int _C, int64_t seed)
    : K(_K), N(_N), C(_C), rand_gen(seed), async_thread(&stream_loader_t::async_process, this) {
}

void stream_loader_t::async_process() {
    while (true) {
    std::unique_lock<std::mutex> lock(request_mutex);
    while (request_queue.empty())
    request_cond.wait(lock);
    auto batch = request_queue.front();
    request_queue.pop_front();
    lock.unlock();

    // an empty batch is a signal for shutdown
    if (!batch.samples.defined())
        break;
    int batch_size = batch.samples.sizes()[0];
    assert(batch.labels.dim() == 1 && batch_size == batch.labels.sizes()[0]
           && batch.aug_samples.dim() > 0 && batch.aug_labels.dim() == 1);
    int R = batch.aug_samples.sizes()[0] - batch_size;
    std::uniform_int_distribution<unsigned int> dice(0, N + R);
    assert(R > 0 && R + batch_size == batch.aug_labels.sizes()[0]
           && R + batch_size == batch.aug_weights.sizes()[0]);

    // initialization of the augmented result
    for (int i = 0; i < batch_size; i++) {
        batch.aug_samples.index_put_({i}, batch.samples[i]);
        batch.aug_labels.index_put_({i}, batch.labels[i]);
        batch.aug_weights.index_put_({i}, 1.0);
    }

    // selection without replacement
    // get_samples
    std::unordered_map<int, std::unordered_set<int>> choices;
    int i = batch_size, choices_size = 0;
    while (i < batch_size + R && rehearsal_size - choices_size > 0) {
        auto map_it = rehearsal_map.begin();
        std::advance(map_it, dice(rand_gen) % rehearsal_map.size());
        if (map_it->second.second.empty())
        continue;
        int index = dice(rand_gen) % map_it->second.second.size();
        auto &set = choices[map_it->first];
        auto set_it = set.find(index);
        if (set_it != set.end())
        continue;
        set.emplace_hint(set_it, index);
        choices_size++;
        batch.aug_samples.index_put_({i}, map_it->second.second[index]);
        batch.aug_labels.index_put_({i}, map_it->first);
        batch.aug_weights.index_put_({i}, map_it->second.first);
        i++;
    }
    batch.aug_size = i;
    lock.lock();
    response_queue.emplace_back(batch);
    lock.unlock();
    request_cond.notify_one();

    // update the rehearsal buffer
    // accumulate
    for (int i = 0; i < batch_size; i++) {
        auto label = batch.labels[i].item<int>();
        auto &buffer = rehearsal_map[label].second;
        if (buffer.size() < N) {
        buffer.emplace_back(batch.samples.index({i}));
        rehearsal_size++;
        } else {
        unsigned int index = dice(rand_gen);
        if (index < N)
            buffer[index] = batch.samples.index({i});
        }
        counts[label] += 1;
    }

    // update weight
    double weight = (double) batch_size / (double) (R * rehearsal_size);
    for (auto& map_it : rehearsal_map) {
    map_it.second.first = std::max(std::log(counts[map_it.first] * weight), 1.0);
    }
    }
}

void stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &labels,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_labels, const torch::Tensor &aug_weights) {
    std::unique_lock<std::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
    request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, labels, aug_samples, aug_labels, aug_weights));
    lock.unlock();
    request_cond.notify_one();
}

int stream_loader_t::wait() {
    std::unique_lock<std::mutex> lock(request_mutex);
    while (response_queue.empty())
    request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return batch.aug_size;
}

size_t stream_loader_t::get_rehearsal_size() {
    return rehearsal_size;
}

stream_loader_t::~stream_loader_t() {
    std::unique_lock<std::mutex> lock(request_mutex);
    request_queue.push_back(queue_item_t());
    lock.unlock();
    request_cond.notify_one();
    async_thread.join();
}
