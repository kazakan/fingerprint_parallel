#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "ImgStatics.hpp"
#include "ImgTransform.hpp"
#include "OclInfo.hpp"

#define BYTE uint8_t

TEST(ImageTransformTest, Negate) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    vector<BYTE> vOriginal(256);  // 0,1,2, ... ,255
    vector<BYTE> vExpected(256);  // 255,254, ... , 0

    for (int i = 0; i < 256; ++i) {
        vOriginal[i] = i;
        vExpected[i] = 255 - i;
    }

    MatrixBuffer<BYTE> bufferOrininal(vOriginal);
    MatrixBuffer<BYTE> bufferNegated(1, 256);

    MatrixBuffer<BYTE> bufferExpected(vExpected);

    bufferOrininal.createBuffer(oclInfo.ctx);
    bufferNegated.createBuffer(oclInfo.ctx);
    bufferOrininal.toGpu(oclInfo);

    imgTransformer.negate(bufferOrininal, bufferNegated);
    bufferNegated.toHost(oclInfo);

    EXPECT_EQ(bufferNegated, bufferExpected);
}

TEST(ImageTransformTest, Copy) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<BYTE> bufferOriginal({1, 50, 126, 200, 255});
    MatrixBuffer<BYTE> bufferCopied(1, 5);

    bufferOriginal.createBuffer(oclInfo.ctx);
    bufferCopied.createBuffer(oclInfo.ctx);

    bufferOriginal.toGpu(oclInfo);

    imgTransformer.copy(bufferOriginal, bufferCopied);
    bufferCopied.toHost(oclInfo);

    EXPECT_EQ(bufferCopied, bufferOriginal);
}

TEST(ImageTransformTest, Normalize) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);

    // 0: M0, 1: V0, 2: width, 3: height, 4: original data, 5: expected result
    using normalize_datatype =
        std::tuple<int, int, int, int, vector<BYTE>, vector<BYTE>>;

    vector<normalize_datatype> datasets{
        {128,
         2000,
         3,
         3,
         {237, 163, 52, 65, 129, 218, 62, 148, 212},
         {190, 141, 67, 76, 118, 177, 74, 131, 173}},
        {128,
         2000,
         3,
         3,
         {237, 163, 52, 65, 129, 218, 62, 148, 212},
         {190, 141, 67, 76, 118, 177, 74, 131, 173}}};

    auto test_one_pair = [&](normalize_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<2>(data), std::get<3>(data),
                                          std::get<4>(data));
        MatrixBuffer<BYTE> bufferResult(std::get<2>(data), std::get<3>(data));
        MatrixBuffer<BYTE> bufferExpected(std::get<2>(data), std::get<3>(data),
                                          std::get<5>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        float mean = imgStatics.mean(bufferOriginal);
        float var = imgStatics.var(bufferOriginal);

        imgTransformer.normalize(bufferOriginal, bufferResult,
                                 std::get<0>(data), std::get<1>(data), mean,
                                 var);

        bufferResult.toHost(oclInfo);

        EXPECT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, DynamicThresholding) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    // 0: blockSize, 1: scale, 2: width, 3: height, 4: original data, 5:
    // expected result
    using dynamic_thresholding_datatype =
        std::tuple<int, float, int, int, vector<BYTE>, vector<BYTE>>;

    vector<dynamic_thresholding_datatype> datasets{
        {3,
         1.05,
         7,
         5,
         {0, 0,   0,   0,   0,   0,   0,  //
          0, 100, 100, 100, 100, 100, 0,  //
          0, 100, 100, 100, 100, 100, 0,  //
          0, 100, 100, 100, 100, 100, 0,  //
          0, 0,   0,   0,   0,   0,   0},
         {0, 0, 0, 0, 0, 0, 0,  //
          0, 1, 1, 1, 1, 1, 0,  //
          0, 1, 0, 0, 0, 1, 0,  //
          0, 1, 1, 1, 1, 1, 0,  //
          0, 0, 0, 0, 0, 0, 0}}};

    auto test_one_pair = [&](dynamic_thresholding_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<2>(data), std::get<3>(data),
                                          std::get<4>(data));
        MatrixBuffer<BYTE> bufferResult(std::get<2>(data), std::get<3>(data));
        MatrixBuffer<BYTE> bufferExpected(std::get<2>(data), std::get<3>(data),
                                          std::get<5>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.applyDynamicThresholding(
            bufferOriginal, bufferResult, std::get<0>(data), std::get<1>(data));

        bufferResult.toHost(oclInfo);

        EXPECT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}

TEST(ImageTransformTest, ApplyGaussian) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    //  0: width, 1: height, 2: original data, 3: expected result
    using gaussian_datatype = std::tuple<int, int, vector<BYTE>, vector<BYTE>>;

    vector<gaussian_datatype> datasets{{5,
                                        5,
                                        {0, 0, 0,   0, 0,  //
                                         0, 0, 0,   0, 0,  //
                                         0, 0, 100, 0, 0,  //
                                         0, 0, 0,   0, 0,  //
                                         0, 0, 0,   0, 0},
                                        {0, 0,  0,  0,  0,  //
                                         0, 6,  13, 6,  0,  //
                                         0, 13, 25, 13, 0,  //
                                         0, 6,  13, 6,  0,  //
                                         0, 0,  0,  0,  0}}};

    auto test_one_pair = [&](gaussian_datatype& data) {
        MatrixBuffer<BYTE> bufferOriginal(std::get<0>(data), std::get<1>(data),
                                          std::get<2>(data));
        MatrixBuffer<BYTE> bufferResult(std::get<0>(data), std::get<1>(data));
        MatrixBuffer<BYTE> bufferExpected(std::get<0>(data), std::get<1>(data),
                                          std::get<3>(data));

        bufferOriginal.createBuffer(oclInfo.ctx);
        bufferResult.createBuffer(oclInfo.ctx);
        bufferOriginal.toGpu(oclInfo);

        imgTransformer.applyGaussianFilter(
            bufferOriginal, bufferResult);

        bufferResult.toHost(oclInfo);

        EXPECT_EQ(bufferResult, bufferExpected);
    };

    for (auto& data : datasets) {
        test_one_pair(data);
    }
}
