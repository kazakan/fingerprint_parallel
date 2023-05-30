#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class ImgStatics {
  private:
    OclInfo oclInfo;
    cl::Program program;

  public:
    ImgStatics(OclInfo oclInfo, string source);
    void sum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
    void mean(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
    void var(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst);
};