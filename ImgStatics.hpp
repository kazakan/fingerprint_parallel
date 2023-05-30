#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

class ImgStatics{
    public:
    void sum(MatrixBuffer& src, MatrixBuffer& dst);
    void mean(MatrixBuffer& src, MatrixBuffer& dst);
    void var(MatrixBuffer& src, MatrixBuffer& dst);
};