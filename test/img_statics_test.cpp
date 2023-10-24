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
    MatrixBuffer<BYTE> bufferOrininal(1, vData.size(), vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    int expected = std::accumulate(vData.begin(), vData.end(), 0);
    float result = imgStatics.sum(bufferOrininal);

    EXPECT_EQ(result, expected);

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

        EXPECT_EQ(result, expected);
    }
}

TEST(ImgStaticsTest, Mean) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgStatics imgStatics(oclInfo);

    vector<BYTE> vData{1, 4, 6, 8, 9};
    MatrixBuffer<BYTE> bufferOrininal(1, vData.size(), vData);
    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    float expected =
        static_cast<float>(std::accumulate(vData.begin(), vData.end(), 0)) /
        vData.size();
    float result = imgStatics.mean(bufferOrininal);

    EXPECT_EQ(result, expected);

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

        EXPECT_EQ(result, expected);
    }
}
