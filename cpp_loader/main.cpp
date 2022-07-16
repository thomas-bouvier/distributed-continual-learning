#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include "distributed_stream_loader.hpp"

unsigned int K = 5;
unsigned int N = 10;
unsigned int R = 2;
unsigned int C = 2;
int64_t seed = 42;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_id> [.. <address> <provider_id>]" << std::endl;
        exit(0);
    }
    uint16_t server_id = atoi(argv[1]);
    std::vector<std::pair<std::string, int>> endpoints;
    for (int i = 2; i < argc; i += 2) {
        auto endpoint = std::make_pair<>(std::string(argv[i]), atoi(argv[i + 1]));
        endpoints.push_back(endpoint);
        std::cout << "Endpoint " << endpoint.first << ", " << endpoint.second << std::endl;
    }
    while (true) {
        std::string address;
        int provider_id;
        std::cout << "Additional endpoint? ";
        std::cin >> address;
        if (address == "no") break;
        std::cin >> provider_id;
        auto endpoint = std::make_pair<>(address, provider_id);
        endpoints.push_back(endpoint);
        std::cout << "Endpoint " << address << ", " << provider_id << std::endl;
        std::cin.clear();
    }
    distributed_stream_loader_t sl(K, N, C, seed, server_id, endpoints);

    torch::Tensor aug_samples = torch::zeros({N + R, 3, 224, 224});
    torch::Tensor aug_labels = torch::randint(K, {N + R});
    torch::Tensor aug_weights = torch::zeros({N + R});

    auto random_batch = [server_id]() -> std::tuple<torch::Tensor, torch::Tensor> {
        torch::Tensor labels = torch::randint(K, {N});
        torch::Tensor samples = torch::full({N, 3, 224, 224}, server_id);
        return std::make_tuple<>(samples, labels);
    };

    std::cout << "Round 1" << std::endl;
    auto batch = random_batch();
    sl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    //int size = sl.wait();
    //std::cout << "size: " << size << std::endl;
    //std::cout << size << ", " << std::get<0>(batch) << ", " << aug_samples << std::endl;

    std::cout << "Round 2" << std::endl;
    batch = random_batch();
    sl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    //size = sl.wait();
    //std::cout << "size: " << size << std::endl;
    //std::cout << size << ", " << std::get<0>(batch) << ", " << aug_samples << std::endl;

    for (int i = 3; i < 1000; i++) {
        std::cout << "Round " << i << std::endl;
        batch = random_batch();
        sl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
        //size = sl.wait();
        //std::cout << "size: " << size << std::endl;
    }

    return 0;
}
