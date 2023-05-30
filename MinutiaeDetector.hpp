#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class MinutiaeDetector {
  private:
    OclInfo oclInfo;
    cl::Program program;

  public:
    MinutiaeDetector(OclInfo oclInfo, string source);
    void applyCrossNumber(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
};