
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>

#include "MatrixBuffer.hpp"

class RandomMatrixGenerator {
    const int SEED = 47;

   public:
    std::random_device rd;
    std::mt19937_64 gen;

    RandomMatrixGenerator() : gen(47) {}

    std::tuple<int, int, std::vector<uint8_t>> generateMatData(
        int minValue, int maxValue, int width = -1, int height = -1) {
        std::uniform_int_distribution<int> sizeDis(4, 1024);

        std::uniform_int_distribution<int> valueDist(minValue, maxValue);

        if (width == -1) {
            width = sizeDis(gen);
        }

        if (height == -1) {
            height = sizeDis(gen);
        }

        const int len = width * height;
        std::vector<uint8_t> arr(len);

        for (int i = 0; i < len; ++i) {
            arr[i] = valueDist(gen);
        }

        return {width, height, arr};
    }
};