#pragma once

#include <iostream>

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"
#include "ScalarBuffer.hpp"

namespace fingerprint_parallel {
namespace core {

/**
 * @brief Class contains operations about ImageTransform.
 */
class ImgTransform {
   private:
    OclInfo ocl_info;
    cl::Program program;

    /**
     * @brief  One interation behavior of rosenfield 4 connected thinnining
     * algorithm.
     * @param src Input buffer
     * @param dst Output buffer
     * @param dir Border direction to calculate. (N,E,S,W) = (0,1,2,3)
     * @return Whether none of any pixel changed.
     */
    bool thinning_one_iter(MatrixBuffer<uint8_t> &src,
                           MatrixBuffer<uint8_t> &dst, int dir);

    /**
     * @brief  One interation behavior of rosenfield 8 connected thinnining
     * algorithm.
     * @param src Input buffer
     * @param dst Output buffer
     * @param dir Border direction to calculate. (N,E,S,W) = (0,1,2,3)
     * @return Whether none of any pixel changed.
     */
    bool thinning8_one_iter(MatrixBuffer<uint8_t> &src,
                            MatrixBuffer<uint8_t> &dst, int dir);

   public:
    ImgTransform(OclInfo ocl_info);

    /**
     * @brief Get cl::Image2D as input, transform it to grayscale.
     *        Result will be returned as MatrixBuffer<uint8_t> which represents
     *        One channel 2D image.
     * @param src Image to transform.
     * @param dst MatrixBuffer<uint8_t> where result to be saved
     */
    void to_gray_scale(cl::Image2D &src, MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Negate image. Simply performed by 255 - pixel.
     * @param src Original image.
     * @param dst Where negated image saved.
     */
    void negate(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Normalize image. M0 +- sqrt(V0*(x-M)^2/V).
     * @param src Original image.
     * @param dst Where normalized image saved.
     * @param M0 Mean after normalized.
     * @param V0 Variance after normalized.
     * @param M Original image mean.
     * @param V Original image variance.
     */
    void normalize(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst,
                   float M0, float V0, ScalarBuffer<float> &M,
                   ScalarBuffer<float> &V);

    /**
     * @brief Binarize image. If pixel > threshold then 255
     * else 0;
     * @param src Original image.
     * @param dst Where result be saved.
     * @param threshold Threshol value.
     */
    void binarize(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst,
                  int threshold = 125);

    /**
     * @brief Dynamic thresholding method. If pixel > avg(block pixels) then 255
     * else 0;
     * @param src Original image.
     * @param dst Where result be saved.
     * @param block_size One side length of block.
     * @param scale scale factor of threshold. Threshold is mean*scale. Default
     * = 1.05
     */
    void dynamic_thresholding(MatrixBuffer<uint8_t> &src,
                              MatrixBuffer<uint8_t> &dst, int block_size,
                              float scale = 1.05);

    /**
     * @brief Apply Rosenfield 4 connectivity thinning algorithm.
     * @param src Original image.
     * @param dst Where result be saved.
     */
    void thinning(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Apply Rosenfield 8 connectivity thinning algorithm.
     * @param src Original image.
     * @param dst Where result be saved.
     */
    void thinning8(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Apply 3x3 Gaussian filter.
     * @param src Original image.
     * @param dst Where result be saved.
     */
    void gaussian_filter(MatrixBuffer<uint8_t> &src,
                         MatrixBuffer<uint8_t> &dst);

    /**
     * @brief Copy image to dst from src.
     * @param src Original image.
     * @param dst Where result be saved.
     */
    void copy(MatrixBuffer<uint8_t> &src, MatrixBuffer<uint8_t> &dst);
};

}  // namespace core
}  // namespace fingerprint_parallel