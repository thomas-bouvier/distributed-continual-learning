#include "distributed_stream_loader.hpp"

#include <unordered_set>
#include <assert.h>

#include <thallium/serialization/stl/unordered_map.hpp>
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <cereal/types/string.hpp>

#define __DEBUG
#include "debug.hpp"

using namespace torch::indexing;


distributed_stream_loader_t::distributed_stream_loader_t(unsigned int _K, unsigned int _N, unsigned int _C,
    int64_t seed, uint16_t server_id, const std::string& server_address,
    const std::vector<std::pair<int, std::string>>& endpoints)
        : tl::provider<distributed_stream_loader_t>([](uint16_t provider_id, const std::string& address) -> tl::engine& {
            static tl::engine myServer(address, THALLIUM_SERVER_MODE);
            std::cout << "Server running at address " << myServer.self()
                << " with provider id " << provider_id << std::endl;
            return myServer;
        }(server_id, server_address), server_id), K(_K), N(_N),
        C(_C), rand_gen(seed) {
    define("get_samples", &distributed_stream_loader_t::get_remote_samples);

    tl::managed<tl::xstream> es = tl::xstream::create();
    xstream = std::move(es);
    tl::managed<tl::thread> thread = xstream->make_thread([this]() {
        async_process();
    });
    async_thread = std::move(thread);
    if (endpoints.size() > 0) {
        add_endpoints(endpoints);
    }
}

void distributed_stream_loader_t::add_endpoints(const std::vector<std::pair<int, std::string>>& endpoints) {
    if (provider_handles.size() == 0) {
        get_samples_procedure = get_engine().define("get_samples");
    }
    provider_handles.reserve(provider_handles.size() + endpoints.size());
    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.second << ", " << endpoint.first << std::endl;
        tl::endpoint server = get_engine().lookup(endpoint.second);
        provider_handles.emplace_back(tl::provider_handle(server, endpoint.first));
    }
}

void distributed_stream_loader_t::async_process() {
    while (true) {
        std::unique_lock<tl::mutex> lock(request_mutex);
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
        assert(R > 0 && R + batch_size == batch.aug_labels.sizes()[0]
            && R + batch_size == batch.aug_weights.sizes()[0]);

        // initialization of the augmented result
        for (int i = 0; i < batch_size; i++) {
            batch.aug_samples.index_put_({i}, batch.samples[i]);
            batch.aug_labels.index_put_({i}, batch.labels[i]);
            batch.aug_weights.index_put_({i}, 1.0);
        }

        // selection without replacement from remote nodes + current node
        // get_samples in Python
        const unsigned int max_global_index = (provider_handles.size() + 1) * K * N;
        std::uniform_int_distribution<unsigned int> dice(0, max_global_index - 1);
        std::vector<unsigned int> choices(R);
        int i = 0;
        while (i < R) {
            int random_global_index = dice(rand_gen);
            if (std::find(choices.begin(), choices.end(), random_global_index) != choices.end())
                continue;
            choices[i++] = random_global_index;
        }

        //TODO async calls dispatching non-blocking requests
        //std::vector<tl::async_response> requests(choices.size());
        int j = batch_size;
        for (size_t i = 0; i < choices.size(); i++) {
            int global_index = choices[i];
            int local_index = global_index % (K * N);
            size_t node = global_index / (K * N);
            assert(node >= 0 && node < (provider_handles.size() + 1));
            // choose remote node
            if (node > 0) {
                size_t remote_index = node - 1;
                tl::provider_handle& ph = provider_handles[remote_index];
                rehearsal_map_t samples = get_samples_procedure.on(ph)(local_index);
                for (const auto& it : samples) {
                    for (const auto& sample : it.second.second) {
                        batch.aug_samples.index_put_({j}, sample);
                        batch.aug_labels.index_put_({j}, it.first);
                        batch.aug_weights.index_put_({j}, it.second.first);
                        j++;
                    }
                }
            } else {
                get_samples(local_index);
                for (const auto& it : selected_samples) {
                    for (const auto& sample : it.second.second) {
                        batch.aug_samples.index_put_({j}, sample);
                        batch.aug_labels.index_put_({j}, it.first);
                        batch.aug_weights.index_put_({j}, it.second.first);
                        j++;
                    }
                }
            }
        }
        batch.aug_size = j;

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();

        // update the rehearsal buffer
        // accumulate in Python
        dice.param(std::uniform_int_distribution<unsigned int>::param_type(0, batch_size - 1));
        for (int i = 0; i < batch_size; i++) {
            if (dice(rand_gen) >= C)
                break;
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
            counts[label]++;
            history_count++;
        }
        // update weight
        double weight = (double) batch_size / (double) (R * rehearsal_size);
        for (auto& map_it : rehearsal_map) {
            map_it.second.first = std::max(std::log(counts[map_it.first] * weight), 1.0);
        }
    }
}

void distributed_stream_loader_t::get_remote_samples(const tl::request& req, unsigned int index) {
    std::cout << "Sending samples to remote node" << std::endl;
    get_samples(index);
    req.respond(selected_samples);
}

// TODO: pass a vector of indices
void distributed_stream_loader_t::get_samples(unsigned int index) {
    if (rehearsal_size == 0)
        return;
    selected_samples.clear();
    size_t rehearsal_class = index / N;
    auto map_it = rehearsal_map.begin();
    std::advance(map_it, rehearsal_class % rehearsal_map.size());
    size_t rehearsal_class_index = index % N % map_it->second.second.size();
    assert(rehearsal_class_index < map_it->second.second.size());
    auto sample = map_it->second.second[rehearsal_class_index];
    auto label = map_it->first;
    auto weight = map_it->second.first;
    buffer_t vec_tensors;
    vec_tensors.push_back(sample);
    selected_samples[label] = std::make_pair(weight, vec_tensors);
}

void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &labels,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_labels, const torch::Tensor &aug_weights) {
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, labels, aug_samples, aug_labels, aug_weights));
    lock.unlock();
    request_cond.notify_one();
}

int distributed_stream_loader_t::wait() {
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (response_queue.empty())
        request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return batch.aug_size;
}

size_t distributed_stream_loader_t::get_rehearsal_size() {
    return rehearsal_size;
}

size_t distributed_stream_loader_t::get_history_count() {
    return history_count;
}

distributed_stream_loader_t::~distributed_stream_loader_t() {
    std::unique_lock<tl::mutex> lock(request_mutex);
    request_queue.push_back(queue_item_t());
    lock.unlock();
    request_cond.notify_one();
    async_thread->join();

    get_engine().wait_for_finalize();
}

namespace cereal {
template<typename A> void save(A& ar, const torch::Tensor& t) {
    std::stringstream ss;
    torch::save(t, ss);
    ar(ss.str());
}

template<typename A> void load(A& ar, torch::Tensor& t) {
    std::string s;
    ar(s);
    std::stringstream ss(s);
    torch::load(t, ss);
}
}
