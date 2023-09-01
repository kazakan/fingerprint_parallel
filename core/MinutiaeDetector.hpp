#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

/**
 * @brief Class extracts minutiaes from preprocessed binary image.
 */
class MinutiaeDetector {
   private:
    OclInfo oclInfo;
    cl::Program program;

   public:
    MinutiaeDetector(OclInfo oclInfo);

    /**
     * @brief Calulates cross numbers per pixel.
     * @param src MatrixBuffer<BYTE> to calculate
     * @param dst MatrixBuffer that Result be saved.
     */
    void applyCrossNumber(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst);
};