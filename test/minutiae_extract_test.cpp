#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>

#include "MinutiaeDetector.hpp"
#include "OclInfo.hpp"
#include "random_case_generator.hpp"

TEST(MinutiaeDetectTest, ApplyCrossNumber) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    MinutiaeDetector detector(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using crossnumber_datatype =
        std::tuple<int, int, std::vector<uint8_t>, std::vector<uint8_t>>;

    std::vector<crossnumber_datatype> datasets{
        // TC1
        {3,
         3,
         {
             0, 0, 0,  //
             0, 1, 0,  //
             0, 0, 0,  //
         },
         {
             0, 0, 0,  //
             0, 0, 0,  //
             0, 0, 0,  //
         }},

        // TC2
        {3,
         3,
         {
             0, 1, 0,  //
             0, 1, 0,  //
             0, 0, 0,  //
         },
         {
             0, 1, 0,  //
             0, 1, 0,  //
             0, 0, 0,  //
         }},
        // TC3
        {3,
         3,
         {
             0, 1, 0,  //
             0, 1, 1,  //
             0, 0, 0,  //
         },
         {
             0, 1, 0,  //
             0, 2, 1,  //
             0, 0, 0,  //
         }},
        // TC4
        {3,
         3,
         {
             0, 1, 0,  //
             1, 1, 1,  //
             0, 0, 0,  //
         },
         {
             0, 1, 0,  //
             1, 3, 1,  //
             0, 0, 0,  //
         }},
        // TC5
        {3,
         3,
         {
             0, 1, 0,  //
             1, 1, 1,  //
             0, 1, 0,  //
         },
         {
             0, 1, 0,  //
             1, 4, 1,  //
             0, 1, 0,  //
         }},
        // TC6
        {3,
         3,
         {
             1, 1, 1,  //
             1, 1, 1,  //
             1, 1, 1,  //
         },
         {
             1, 1, 1,  //
             1, 0, 1,  //
             1, 1, 1,  //
         }},
        // TC7
        {3,
         3,
         {
             1, 1, 0,  //
             1, 1, 0,  //
             0, 0, 0,  //
         },
         {
             1, 1, 0,  //
             1, 1, 0,  //
             0, 0, 0,  //
         }},
        // TC7
        {5,
         5,
         {
             1, 1, 0, 1, 0,  //
             1, 1, 0, 1, 1,  //
             0, 0, 0, 1, 1,  //
             0, 1, 0, 1, 1,  //
             0, 0, 0, 1, 1   //
         },
         {
             1, 1, 0, 1, 0,  //
             1, 1, 0, 2, 1,  //
             0, 0, 0, 1, 1,  //
             0, 0, 0, 1, 1,  //
             0, 0, 0, 1, 1   //
         }},
    };

    // create random data
    RandomMatrixGenerator generator;
    const int nRandomCases = 100;
    for (int i = 0; i < nRandomCases; ++i) {
        std::tuple<int, int, std::vector<uint8_t>> inputData =
            generator.generateMatData(0, 1, 5, 5);

        const int NC = std::get<0>(inputData);
        const int NR = std::get<1>(inputData);
        const std::vector<uint8_t>& arr = std::get<2>(inputData);

        const auto value = [&](int r, int c) -> const uint8_t {
            if (r < 0 || r >= NR || c < 0 || c >= NC) {
                return 0;
            }

            return arr[NC * r + c];
        };

        const int dx[] = {0, -1, -1, -1, 0, 1, 1, 1};
        const int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1};

        const auto cn = [&](int idx) -> const uint8_t {
            const int r = idx / NC;
            const int c = idx % NC;

            if (value(r, c) == 0) return 0;

            uint8_t ret = 0;

            for (int i = 0; i < 8; ++i) {
                if (value(r + dx[i], c + dy[i]) !=
                    value(r + dx[(i + 1) % 8], c + dy[(i + 1) % 8])) {
                    ++ret;
                }
            }

            ret >>= 1;

            return ret;
        };

        std::vector<uint8_t> result(arr.size());

        for (int i = 0; i < arr.size(); ++i) {
            result[i] = cn(i);
        }

        datasets.push_back({NC, NR, arr, result});
    }

    auto test_one_pair = [&](crossnumber_datatype& data) {
        MatrixBuffer<uint8_t> bufferOriginal(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        MatrixBuffer<uint8_t> bufferResult(std::get<0>(data),
                                           std::get<1>(data));
        MatrixBuffer<uint8_t> bufferExpected(
            std::get<0>(data), std::get<1>(data), std::get<3>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        detector.applyCrossNumber(bufferOriginal, bufferResult);

        bufferResult.toHost(oclInfo);

        ASSERT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}
