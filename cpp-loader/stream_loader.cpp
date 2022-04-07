#include "stream_loader.hpp"

#include <set>

stream_loader_t::stream_loader_t(unsigned int _K, unsigned int _N, int64_t seed)
    : K(_K), N(_N), rand_gen(seed), rand_class(0, K-1), rand_sample(0, N-1) {
}

void stream_loader_t::accumulate(const sample_list_t &samples) {
    for (auto &sample : samples) {
	auto &label = std::get<0>(sample);
	auto &tensor = std::get<1>(sample);
	auto &buffer = rehearsal_map[label];
	if (buffer.size() < N)
	    buffer.emplace_back(tensor);
	else
	    buffer[rand_sample(rand_gen)] = tensor;
    }
}

sample_list_t stream_loader_t::get_samples(unsigned int R) {
    sample_list_t result;
    std::unordered_map<std::string, std::unordered_set<unsigned int>> choices;

    unsigned int i = 0;
    while (i < R) {
	auto map_it = rehearsal_map.begin();
	std::advance(map_it, rand_class(rand_gen) % rehearsal_map.size());
	if (map_it->second.empty())
	    continue;
        unsigned int index = rand_sample(rand_gen) % map_it->second.size();
	auto &set = choices[map_it->first];
	auto set_it = set.find(index);
	if (set_it != set.end())
	    continue;
	set.emplace_hint(set_it, index);
	result.emplace_back(map_it->first, map_it->second[index]);
	i++;
    }
    return result;
}
