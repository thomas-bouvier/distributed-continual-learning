#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include "distributed_stream_loader.hpp"

unsigned int K = 10;
unsigned int N = 20;
unsigned int R = 5;
unsigned int C = 5;
int64_t seed = 42;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <server_id> <server_address> [.. <provider_id> <address>]" << std::endl;
        exit(0);
    }
    uint16_t server_id = atoi(argv[1]);
    std::string server_address = argv[2];

    std::vector<std::pair<int, std::string>> endpoints;
    for (int i = 3; i < argc; i += 2) {
        auto endpoint = std::make_pair<>(atoi(argv[i]), std::string(argv[i + 1]));
        endpoints.push_back(endpoint);
        std::cout << "Endpoint " << endpoint.first << ", " << endpoint.second << std::endl;
    }
    distributed_stream_loader_t dsl(K, N, C, seed, server_id, server_address, endpoints);

    endpoints.clear();
    while (true) {
        std::string address;
        int provider_id;
        std::cout << "Additional endpoint? ";
        std::cin >> address;
        if (address == "no") break;
        std::cin >> provider_id;
        auto endpoint = std::make_pair<>(provider_id, address);
        endpoints.push_back(endpoint);
        std::cout << "Endpoint " << address << ", " << provider_id << std::endl;
        std::cin.clear();
    }
    dsl.add_endpoints(endpoints);

    torch::Tensor aug_samples = torch::zeros({N + R, 3, 224, 224});
    torch::Tensor aug_labels = torch::randint(K, {N + R});
    torch::Tensor aug_weights = torch::zeros({N + R});

    auto random_batch = [server_id]() -> std::tuple<torch::Tensor, torch::Tensor> {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor labels = torch::randint(K, {N});
        torch::Tensor samples = torch::full({N, 3, 224, 224}, server_id, options);
        return std::make_tuple<>(samples, labels);
    };

    std::cout << "Round 1" << std::endl;
    auto batch = random_batch();
    dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    int size = dsl.wait();
    std::cout << "size: " << size << std::endl;

    std::cout << "Round 2" << std::endl;
    batch = random_batch();
    dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    size = dsl.wait();
    std::cout << "size: " << size << std::endl;

    for (int i = 3; i < 1000; i++) {
        std::cout << "Round " << i << std::endl;
        batch = random_batch();
        dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
        size = dsl.wait();
        std::cout << "size: " << size << std::endl;
    }

    return 0;
}
