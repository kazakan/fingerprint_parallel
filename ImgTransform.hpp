#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"
#include <iostream>

class ImgTransform {
  private:
    OclInfo oclInfo;
    cl::Program program;
    bool thinningOneIter(MatrixBuffer<BYTE>& src,MatrixBuffer<BYTE>& dst,int dir);
    bool thinning8OneIter(MatrixBuffer<BYTE>& src,MatrixBuffer<BYTE>& dst,int dir);

  public:
    ImgTransform(OclInfo oclInfo, string source);

    void toGrayScale(cl::Image2D &src, MatrixBuffer<BYTE> &dst);
    void negate(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
    void normalize(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, float M0, float V0, float M, float V);
    void applyDynamicThresholding(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int blockSize);
    void applyThinning(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
    void applyThinning8(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);

    void applyGaussianFilter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
};