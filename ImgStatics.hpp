#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class ImgStatics {
  private:
    OclInfo oclInfo;
    cl::Program program;

  public:
    ImgStatics(OclInfo oclInfo, string source);
    float sum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
    float mean(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
    float var(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
    float squareSum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
};