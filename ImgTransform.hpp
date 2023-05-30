#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"

class ImgTransform {
  private:
    OclInfo oclInfo;
    cl::Program program;
    bool thinningOneIter(MatrixBuffer<BYTE>& src,MatrixBuffer<BYTE>& dst);

  public:
    ImgTransform(OclInfo oclInfo, string source);

    void toGrayScale(cl::Image2D &src, MatrixBuffer<BYTE> &dst);
    void normalize(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, float M0, float V0, float M, float V);
    void applyDynamicThresholding(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int blockSize);
    void applyThinning(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);

    void applyGaussianFilter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
};