#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class ImgStatics{
    private:
    OclInfo oclInfo;
    cl::Program program;
    public:
    ImgStatics(OclInfo oclInfo, string source);
    void sum(MatrixBuffer& src, MatrixBuffer& dst);
    void mean(MatrixBuffer& src, MatrixBuffer& dst);
    void var(MatrixBuffer& src, MatrixBuffer& dst);
};