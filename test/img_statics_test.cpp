#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>

#include "ImgStatics.hpp"
#include "OclInfo.hpp"
#include "random_case_generator.hpp"

TEST(ImgStaticsTest, Sum) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, vector<BYTE>, int>;

    vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 45},
        {3, 3, {255, 255, 255, 255, 255, 255, 255, 255, 255}, 2295},
        {3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 0},
        {1, 1, {1}, 1},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        tuple<int, int, vector<BYTE>> inputData =
            generator.generateMatData(0, 255);

        const vector<BYTE>& arr = std::get<2>(inputData);
        const int N = arr.size();

        long long sum = 0;

        for (int i = 0; i < N; ++i) {
            sum += arr[i];
        }

        datasets.push_back(
            {std::get<0>(inputData), std::get<1>(inputData), arr, sum});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<0>(data), std::get<1>(data),
                                          std::get<2>(data));
        double expected = std::get<3>(data);

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        double result = imgStatics.sum(bufferOriginal);

        ASSERT_EQ(result, expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, SqaureSum) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, vector<BYTE>, long long>;

    vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 285},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        tuple<int, int, vector<BYTE>> inputData =
            generator.generateMatData(0, 255);

        const vector<BYTE>& arr = std::get<2>(inputData);
        const int N = arr.size();

        long long sum = 0;

        for (long long v : arr) {
            sum += v * v;
        }

        datasets.push_back(
            {std::get<0>(inputData), std::get<1>(inputData), arr, sum});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<0>(data), std::get<1>(data),
                                          std::get<2>(data));
        double expected = std::get<3>(data);

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        double result = imgStatics.squareSum(bufferOriginal);

        ASSERT_EQ(result, expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, Mean) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using sum_datatype = std::tuple<int, int, vector<BYTE>, float>;

    vector<sum_datatype> datasets{
        {3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 5},
        {3, 3, {255, 255, 255, 255, 255, 255, 255, 255, 255}, 255},
        {3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, 0},
        {1, 1, {1}, 1},
    };

    // Create random data
    RandomMatrixGenerator generator;
    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        tuple<int, int, vector<BYTE>> inputData =
            generator.generateMatData(0, 255);

        const vector<BYTE>& arr = std::get<2>(inputData);
        const int N = arr.size();

        long long sum = 0;

        for (int i = 0; i < N; ++i) {
            sum += arr[i];
        }

        float mean = static_cast<float>(sum) / N;

        datasets.push_back(
            {std::get<0>(inputData), std::get<1>(inputData), arr, mean});
    }

    auto test_one_pair = [&](sum_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<0>(data), std::get<1>(data),
                                          std::get<2>(data));
        double expected = std::get<3>(data);

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        float result = imgStatics.mean(bufferOriginal);

        ASSERT_NEAR(result, expected, 0.0001);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImgStaticsTest, Var) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using var_datatype = std::tuple<int, int, vector<BYTE>, double>;

    vector<var_datatype> datasets{
        {3, 3, {76, 49, 136, 167, 143, 160, 75, 220, 71}, 2884.98765432},
        {3, 3, {102, 174, 55, 135, 45, 115, 40, 216, 40}, 3620.24691358024}};

    // Create random data
    RandomMatrixGenerator generator;
    const int nRandomCases = 100;
    for (int randomCaseNo = 0; randomCaseNo < nRandomCases; ++randomCaseNo) {
        tuple<int, int, vector<BYTE>> inputData =
            generator.generateMatData(0, 255);

        const vector<BYTE>& arr = std::get<2>(inputData);

        long long sum = 0;
        long long squareSum = 0;
        const int N = arr.size();

        for (int i = 0; i < N; ++i) {
            long long v = arr[i];
            sum += v;
            squareSum += v * v;
        }

        double mean = static_cast<double>(sum) / N;
        double expected = static_cast<double>(squareSum) / N - mean * mean;

        datasets.push_back(
            {std::get<0>(inputData), std::get<1>(inputData), arr, expected});
    }

    auto test_one_pair = [&](var_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<0>(data), std::get<1>(data),
                                          std::get<2>(data));

        double expected = std::get<3>(data);

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        double result = imgStatics.var(bufferOriginal);

        ASSERT_NEAR(result, expected, 0.0001);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}