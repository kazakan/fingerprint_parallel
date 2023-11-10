#pragma once

#include <CL/cl_platform.h>

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "ScalarBuffer.hpp"

namespace fingerprint_parallel {
namespace core {

/**
 * @brief Class calculates some mathmetical statics from MatrixBuffer<uint8_t>
 *
 */
class ImgStatics {
   private:
    OclInfo ocl_info;
    cl::Program program;

   public:
    ImgStatics(OclInfo ocl_info);

    /**
     * @brief Sum all the elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     *
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return sum of elements
     */
    void sum(MatrixBuffer<uint8_t> &src, ScalarBuffer<uint64_t> &ret);

    /**
     * @brief Get Sum of x^2 in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Sum of x^2 in elements
     */
    void square_sum(MatrixBuffer<uint8_t> &src, ScalarBuffer<uint64_t> &ret);

    /**
     * @brief Get average of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Average of elements
     */
    void mean(MatrixBuffer<uint8_t> &src, ScalarBuffer<cl_float> &ret);

    /**
     * @brief Get variance of elements in buffer.
     *        This copies result from gpu because needs of aggregation
     *        across work groups.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @return Variance of elements
     */
    void var(MatrixBuffer<uint8_t> &src, ScalarBuffer<cl_float> &ret);
};

}  // namespace core
}  // namespace fingerprint_parallel