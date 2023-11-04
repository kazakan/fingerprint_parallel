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
     * @param src MatrixBuffer<uint8_t> to calculate
     * @param dst MatrixBuffer that Result be saved.
     */
    void applyCrossNumber(MatrixBuffer<uint8_t> &src,
                          MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Calulates cross numbers per pixel.
     * @param src MatrixBuffer<uint8_t> after applyCrossNumber
     * @param dst MatrixBuffer that Result be saved.
     */
    void removeFalseMinutiae(MatrixBuffer<uint8_t> &src,
                             MatrixBuffer<uint8_t> &dst);
};