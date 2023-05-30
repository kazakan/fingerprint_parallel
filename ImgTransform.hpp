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

    void toGrayScale(Img& src, MatrixBuffer& dst);
    void normalize(MatrixBuffer& src, MatrixBuffer& dst);
    void applyGaborFilter(MatrixBuffer& src, MatrixBuffer& dst);
    void applyDynamicThresholding(MatrixBuffer& src, MatrixBuffer& dst);
    void applyThinning(MatrixBuffer& src, MatrixBuffer& dst);
    
    void applyGaussianFilter(MatrixBuffer& src, MatrixBuffer& dst);
};