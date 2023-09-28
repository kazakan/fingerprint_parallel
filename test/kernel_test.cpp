#include <gtest/gtest.h>

#include "ImgTransform.hpp"
#include "OclInfo.hpp"

TEST(ImageTransformTest, Negate) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<BYTE> buffer1(1, 5);
    MatrixBuffer<BYTE> buffer2(1, 5);
    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    buffer1.getData()[0] = 1;
    buffer1.getData()[1] = 50;
    buffer1.getData()[2] = 126;
    buffer1.getData()[3] = 200;
    buffer1.getData()[4] = 255;

    buffer1.toGpu(oclInfo);

    imgTransformer.negate(buffer1, buffer2);
    buffer2.toHost(oclInfo);

    EXPECT_EQ(buffer2.getData()[0], 255 - 1);
    EXPECT_EQ(buffer2.getData()[1], 255 - 50);
    EXPECT_EQ(buffer2.getData()[2], 255 - 126);
    EXPECT_EQ(buffer2.getData()[3], 255 - 200);
    EXPECT_EQ(buffer2.getData()[4], 255 - 255);
}

TEST(ImageTransformTest, Copy) {
    OclInfo oclInfo = OclInfo::initOpenCL();
    ImgTransform imgTransformer(oclInfo);

    MatrixBuffer<BYTE> buffer1(1, 5);
    MatrixBuffer<BYTE> buffer2(1, 5);
    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    buffer1.getData()[0] = 1;
    buffer1.getData()[1] = 50;
    buffer1.getData()[2] = 126;
    buffer1.getData()[3] = 200;
    buffer1.getData()[4] = 255;

    buffer1.toGpu(oclInfo);

    imgTransformer.copy(buffer1, buffer2);
    buffer2.toHost(oclInfo);

    EXPECT_EQ(buffer2.getData()[0], buffer1.getData()[0]);
    EXPECT_EQ(buffer2.getData()[1], buffer1.getData()[1]);
    EXPECT_EQ(buffer2.getData()[2], buffer1.getData()[2]);
    EXPECT_EQ(buffer2.getData()[3], buffer1.getData()[3]);
    EXPECT_EQ(buffer2.getData()[4], buffer1.getData()[4]);
}