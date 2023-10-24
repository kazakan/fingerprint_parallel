#include <gtest/gtest.h>

#include "ImgTransform.hpp"
#include "OclInfo.hpp"

TEST(ImageTransformTest, Negate) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<BYTE> buffer1(1, 5, {1, 50, 126, 200, 255});
    MatrixBuffer<BYTE> buffer2(1, 5);

    MatrixBuffer<BYTE> expected(
        1, 5, {255 - 1, 255 - 50, 255 - 126, 255 - 200, 255 - 255});

    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    buffer1.toGpu(oclInfo);

    imgTransformer.negate(buffer1, buffer2);
    buffer2.toHost(oclInfo);

    EXPECT_EQ(buffer2, expected);
}

TEST(ImageTransformTest, Copy) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<BYTE> buffer1(1, 5, {1, 50, 126, 200, 255});
    MatrixBuffer<BYTE> buffer2(1, 5);
    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    buffer1.toGpu(oclInfo);

    imgTransformer.copy(buffer1, buffer2);
    buffer2.toHost(oclInfo);

    EXPECT_EQ(buffer2, buffer1);
}