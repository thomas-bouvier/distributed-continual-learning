#include "stream_loader.hpp"

void stream_loader_t::accumulate(const sample_list_t &samples) {
    for (auto &sample : samples)
	std::cout << std::get<0>(sample) << ":" << std::get<1>(sample) << std::endl;	
}
