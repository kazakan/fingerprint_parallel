#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

/**
 * @brief Class calculates some mathmetical statics from MatrixBuffer<uint8_t>
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
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return sum of elements
     */
    std::int64_t sum(MatrixBuffer<uint8_t> &src);

    /**
     * @brief Get average of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Average of elements
     */
    double mean(MatrixBuffer<uint8_t> &src);

    /**
     * @brief Get variance of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Variance of elements
     */
    double var(MatrixBuffer<uint8_t> &src);

    /**
     * @brief Get Sum of x^2 in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Sum of x^2 in elements
     */
    std::int64_t squareSum(MatrixBuffer<uint8_t> &src);
};