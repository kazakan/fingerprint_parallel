#include <gtest/gtest.h>

#include <algorithm>

#include "ImgTransform.hpp"
#include "OclInfo.hpp"

TEST(ImageTransformTest, Negate) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    vector<BYTE> vOriginal(256);  // 0,1,2, ... ,255
    vector<BYTE> vExpected(256);  // 255,254, ... , 0

    for (int i = 0; i < 256; ++i) {
        vOriginal[i] = i;
        vExpected[i] = 255 - i;
    }

    MatrixBuffer<BYTE> bufferOrininal(1, vOriginal.size(), vOriginal);
    MatrixBuffer<BYTE> bufferNegated(1, 256);

    MatrixBuffer<BYTE> bufferExpected(1, vOriginal.size(), vExpected);

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

    MatrixBuffer<BYTE> bufferOriginal(1, 5, {1, 50, 126, 200, 255});
    MatrixBuffer<BYTE> bufferCopied(1, 5);

    bufferOriginal.createBuffer(oclInfo.ctx);
    bufferCopied.createBuffer(oclInfo.ctx);

    bufferOriginal.toGpu(oclInfo);

    imgTransformer.copy(bufferOriginal, bufferCopied);
    bufferCopied.toHost(oclInfo);

    EXPECT_EQ(bufferCopied, bufferOriginal);
}

