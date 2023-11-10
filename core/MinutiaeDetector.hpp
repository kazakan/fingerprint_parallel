#pragma once

#include "Img.hpp"
#include "MatrixBuffer.hpp"

namespace fingerprint_parallel {
namespace core {

/**
 * @brief Class extracts minutiaes from preprocessed binary image.
 */
class MinutiaeDetector {
   private:
    OclInfo ocl_info_;
    cl::Program program_;

   public:
    MinutiaeDetector(OclInfo ocl_info);

    /**
     * @brief Calulates cross numbers per pixel.
     * @param src MatrixBuffer<uint8_t> to calculate
     * @param dst MatrixBuffer that Result be saved.
     */
    void apply_cross_number(MatrixBuffer<uint8_t> &src,
                            MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Calulates cross numbers per pixel.
     * @param src MatrixBuffer<uint8_t> after applyCrossNumber
     * @param dst MatrixBuffer that Result be saved.
     */
    void remove_false_minutiae(MatrixBuffer<uint8_t> &src,
                               MatrixBuffer<uint8_t> &dst);
};

}  // namespace core
}  // namespace fingerprint_parallel