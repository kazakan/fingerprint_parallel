#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

/**
 * @brief Class calculates some mathmetical statics from MatrixBuffer<BYTE>
 *
 */
class ImgStatics {
   private:
    OclInfo oclInfo;
    cl::Program program;

   public:
    ImgStatics(OclInfo oclInfo);

    /**
     * @brief Sum all the elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     *
     * @param src MatrixBuffer<BYTE> to calculate
     * @return sum of elements
     */
    float sum(MatrixBuffer<BYTE> &src);

    /**
     * @brief Get average of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<BYTE> to calculate
     * @return Average of elements
     */
    float mean(MatrixBuffer<BYTE> &src);

    /**
     * @brief Get variance of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<BYTE> to calculate
     * @return Variance of elements
     */
    float var(MatrixBuffer<BYTE> &src);

    /**
     * @brief Get Sum of x^2 in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<BYTE> to calculate
     * @return Sum of x^2 in elements
     */
    float squareSum(MatrixBuffer<BYTE> &src);
};