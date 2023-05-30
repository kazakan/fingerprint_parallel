#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class ImgStatics {
  private:
    OclInfo oclInfo;
    cl::Program program;

  public:
    ImgStatics(OclInfo oclInfo, string source);
    float sum(MatrixBuffer<BYTE> &src);
    float mean(MatrixBuffer<BYTE> &src);
    float var(MatrixBuffer<BYTE> &src);
    float squareSum(MatrixBuffer<BYTE> &src);
};