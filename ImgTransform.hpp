#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "OclInfo.hpp"
#include "OclException.hpp"

class ImgTransform{
    private:
    OclInfo oclInfo;
    cl::Program program;
    public:
    ImgTransform(OclInfo oclInfo, string source);

    void toGrayScale(cl::Image2D& src, MatrixBuffer<BYTE>& dst);
    void normalize(MatrixBuffer<BYTE>& src, MatrixBuffer<BYTE>& dst,float M0, float V0, float M,float V);
    void applyDynamicThresholding(MatrixBuffer<BYTE>& src, MatrixBuffer<BYTE>& dst);
    void applyThinning(MatrixBuffer<BYTE>& src, MatrixBuffer<BYTE>& dst);
    
    void applyGaussianFilter(MatrixBuffer<BYTE>& src, MatrixBuffer<BYTE>& dst);
};