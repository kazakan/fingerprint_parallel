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

    vector<BYTE> vData{1, 4, 6, 8, 9};
    MatrixBuffer<BYTE> bufferOrininal(vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    int expected = std::accumulate(vData.begin(), vData.end(), 0);
    long long result = imgStatics.sum(bufferOrininal);

    ASSERT_EQ(result, expected);

    // Random generated test cases
    const int nCases = 100;
    RandomMatrixGenerator matGen;
    for (int currentCase = 0; currentCase < nCases; ++currentCase) {
        unique_ptr<MatrixBuffer<BYTE>> original = matGen.generateMat(0, 255);
        original->createBuffer(oclInfo.ctx);
        original->toGpu(oclInfo);

        expected = std::accumulate(original->getData(),
                                   original->getData() + original->getLen(), 0);
        result = imgStatics.sum(*original);

        ASSERT_EQ(result, expected);
    }
}

TEST(ImgStaticsTest, SquareSum) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    vector<BYTE> vData{1, 4, 6, 8, 9};
    MatrixBuffer<BYTE> bufferOrininal(vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    long long expected = 0;
    for (long long v : vData) {
        expected += (v * v);
    }
    long long result = imgStatics.squareSum(bufferOrininal);

    ASSERT_EQ(result, expected);

    // Random generated test cases
    const int nCases = 100;
    RandomMatrixGenerator matGen;
    for (int currentCase = 0; currentCase < nCases; ++currentCase) {
        unique_ptr<MatrixBuffer<BYTE>> original = matGen.generateMat(0, 10);
        original->createBuffer(oclInfo.ctx);
        original->toGpu(oclInfo);

        expected = 0;
        for (int i = 0; i < original->getLen(); ++i) {
            const long long v = static_cast<double>(original->getData()[i]);
            expected += (v * v);
        }
        result = imgStatics.squareSum(*original);

        ASSERT_EQ(result, expected);
    }
}

TEST(ImgStaticsTest, Mean) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    vector<BYTE> vData{1, 4, 6, 8, 9};
    MatrixBuffer<BYTE> bufferOrininal(vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    double expected =
        static_cast<double>(std::accumulate(vData.begin(), vData.end(), 0)) /
        vData.size();
    double result = imgStatics.mean(bufferOrininal);

    ASSERT_EQ(result, expected);

    // Random generated test cases
    const int nCases = 100;
    RandomMatrixGenerator matGen;
    for (int currentCase = 0; currentCase < nCases; ++currentCase) {
        unique_ptr<MatrixBuffer<BYTE>> original = matGen.generateMat(0, 255);
        original->createBuffer(oclInfo.ctx);
        original->toGpu(oclInfo);

        expected = static_cast<double>(std::accumulate(
                       original->getData(),
                       original->getData() + original->getLen(), 0)) /
                   original->getLen();
        result = imgStatics.mean(*original);

        ASSERT_EQ(result, expected);
    }
}

TEST(ImgStaticsTest, Var) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using var_datatype = std::tuple<int, int, vector<BYTE>, double>;

    vector<var_datatype> datasets;

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

        ASSERT_EQ(result, expected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}