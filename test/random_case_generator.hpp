
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>

#include "MatrixBuffer.hpp"

using namespace fingerprint_parallel::core;

class RandomMatrixGenerator {
    const int SEED = 47;

   public:
    std::random_device rd_;
    std::mt19937_64 gen_;

    RandomMatrixGenerator() : gen_(47) {}

    std::tuple<int, int, std::vector<uint8_t>> generate_matrix_data(
        int minValue, int maxValue, int width = -1, int height = -1) {
        std::uniform_int_distribution<int> size_dist(4, 1024);

        std::uniform_int_distribution<int> value_dist(minValue, maxValue);

        if (width == -1) {
            width = size_dist(gen_);
        }

        if (height == -1) {
            height = size_dist(gen_);
        }

        const int len = width * height;
        std::vector<uint8_t> arr(len);

        for (int i = 0; i < len; ++i) {
            arr[i] = value_dist(gen_);
        }

        return {width, height, arr};
    }
};