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
    float result = imgStatics.sum(bufferOrininal);

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

    float expected = 0;
    for (float v : vData) {
        expected += v * v;
    }
    float result = imgStatics.squareSum(bufferOrininal);

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
            const float v = static_cast<float>(original->getData()[i]);
            expected +=  v*v;
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

    float expected =
        static_cast<float>(std::accumulate(vData.begin(), vData.end(), 0)) /
        vData.size();
    float result = imgStatics.mean(bufferOrininal);

    ASSERT_EQ(result, expected);

    // Random generated test cases
    const int nCases = 100;
    RandomMatrixGenerator matGen;
    for (int currentCase = 0; currentCase < nCases; ++currentCase) {
        unique_ptr<MatrixBuffer<BYTE>> original = matGen.generateMat(0, 255);
        original->createBuffer(oclInfo.ctx);
        original->toGpu(oclInfo);

        expected = static_cast<float>(std::accumulate(
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

    vector<BYTE> vData{1, 4, 6, 8, 9};
    MatrixBuffer<BYTE> bufferOrininal(1, vData.size(), vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    float mean =
        static_cast<float>(std::accumulate(vData.begin(), vData.end(), 0)) /
        vData.size();
    float squareSum = 0;
    for (int i = 0; i < vData.size(); ++i) {
        squareSum += static_cast<float>(vData[i]) * vData[i];
    }

    float expected = squareSum / vData.size() - mean * mean;
    float result = imgStatics.var(bufferOrininal);

    ASSERT_EQ(result, expected);

    // Random generated test cases
    const int nCases = 100;
    RandomMatrixGenerator matGen;
    for (int currentCase = 0; currentCase < nCases; ++currentCase) {
        unique_ptr<MatrixBuffer<BYTE>> original = matGen.generateMat(0, 255);
        original->createBuffer(oclInfo.ctx);
        original->toGpu(oclInfo);

        mean = static_cast<float>(std::accumulate(
                   original->getData(),
                   original->getData() + original->getLen(), 0)) /
               original->getLen();
        squareSum = 0;
        for (int i = 0; i < original->getLen(); ++i) {
            squareSum += static_cast<float>(original->getData()[i]) *
                         original->getData()[i];
        }

        expected = squareSum / original->getLen() - mean * mean;
        result = imgStatics.var(*original);

        ASSERT_EQ(result, expected);
    }
}