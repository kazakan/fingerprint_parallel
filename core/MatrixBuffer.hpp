#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "CL/opencl.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"

namespace fingerprint_parallel {
namespace core {

/**
 * @brief Class represents Matrix. May contains related Opencl Buffer object.
 *
 */
template <typename T>
class MatrixBuffer {
   protected:
    T *data_ = nullptr;
    std::size_t width_;
    std::size_t height_;
    std::size_t size_;
    cl::Buffer *buffer_ = nullptr;
    OclInfo *ocl_info_ = nullptr;

   public:
    /**
     * @brief Create MatruxBuffer with width, and height.
     * @param width width of image. Number of pixels in row.
     * @param height height of image. Number of pixels in column.
     */
    MatrixBuffer(std::size_t width, std::size_t height)
        : width_(width), height_(height), size_(width * height) {
        data_ = new T[size_];
    };

    /**
     * @brief Create MatruxBuffer with width, and height.
     * @param data initial data
     */
    MatrixBuffer(std::vector<T> data) : MatrixBuffer<T>(1, data.size(), data){};

    /**
     * @brief Create MatruxBuffer with width, and height.
     * @param width width of image. Number of pixels in row.
     * @param height height of image. Number of pixels in column.
     * @param data initial data
     */
    MatrixBuffer(std::size_t width, std::size_t height, std::vector<T> data)
        : width_(width), height_(height), size_(width * height) {
        data_ = new T[size_];

        const std::size_t limit =
            std::min(size_, static_cast<std::size_t>(data.size()));
        for (int i = 0; i < limit; ++i) {
            data_[i] = data[i];
        }
    };

    ~MatrixBuffer() {
        delete[] data_;
        data_ = nullptr;
        if (buffer_ != nullptr) delete buffer_;
    }

    bool operator==(const MatrixBuffer<T> &rhs) const {
        if (width_ != rhs.width_ || height_ != rhs.height_ ||
            size_ != rhs.size_)
            return false;

        for (int i = 0; i < size_; ++i) {
            if (data_[i] != rhs.data_[i]) return false;
        }
        return true;
    }

    /**
     * @brief Initialize OpenCL Buffer for matrix.
     * @param ctx cl::Context object
     * @param memFlag flag used for OpenCL memory access policy. Default =
     * CL_MEM_READ_WRITE
     */
    void create_buffer(OclInfo *ocl_info,
                       cl_mem_flags mem_flag = CL_MEM_READ_WRITE) {
        cl_int err = CL_SUCCESS;

        ocl_info_ = ocl_info;

        if (buffer_ != nullptr) {
            delete buffer_;
        }

        buffer_ = new cl::Buffer(ocl_info_->ctx_, mem_flag, size_ * sizeof(T),
                                 nullptr, &err);

        if (err != CL_SUCCESS) {
            throw OclException(
                "Error while creating OCL buffer in MatrixBuffer.", err);
        }
    }

    /**
     * @return Width of matrix
     */
    const std::size_t width() const { return width_; }

    /**
     * @brief Get height of matrix.
     * @return Height of matrix
     */
    const std::size_t height() const { return height_; }

    /**
     * @brief Get numbers of elements in matrix. (width*height)
     * @return width*height
     */
    const std::size_t size() const { return size_; }

    /**
     * @brief  Get pointer that points first element of matrix.
     * @return Pointer to first element.
     */
    T *data() { return data_; }

    /**
     * @brief Get Related OpenCL Buffer.
     * @return Related OpenCl Buffer
     */
    cl::Buffer *buffer() { return buffer_; }

    OclInfo *ocl_info() { return ocl_info_; }

    /**
     * @brief Copy Host memory to Gpu.
     * @param ocl_info OclInfo contains valid queue.
     * @param blocking if false, only enqueue job and continue. Default=true.
     */
    void to_gpu(bool blocking = true) {
        if (ocl_info_ == nullptr) {
            throw std::runtime_error("ocl_info is not nullptr.");
        }

        cl_int err = ocl_info_->queue_.enqueueWriteBuffer(
            *buffer(), blocking, 0, size() * sizeof(T), (void *)data(), nullptr,
            nullptr);
        if (err) throw OclException("Error enqueueWriteBuffer", err);
    }

    /**
     * @brief Copy Gpu memory to host.
     * @param ocl_info OclInfo contains valid queue.
     * @param blocking if false, only enqueue job and continue. Default=true.
     */
    void to_host(bool blocking = true) {
        if (ocl_info_ == nullptr) {
            throw std::runtime_error("ocl_info is not nullptr.");
        }

        cl_int err = ocl_info_->queue_.enqueueReadBuffer(
            *buffer(), blocking, 0, size() * sizeof(T), (void *)data(), nullptr,
            nullptr);
        if (err) throw OclException("Error enqueueReadBuffer", err);
    }

    /**
     * @brief Use enqueueCopyBuffer to copy buffer in gpu.
     * @param ocl_info OclInfo contains valid queue.
     * @param dst destination to be copied.
     */
    void copy_buffer(MatrixBuffer &dst) {
        if (ocl_info_ == nullptr) {
            throw std::runtime_error("ocl_info is not nullptr.");
        }
        cl_int err = ocl_info_->queue_.enqueueCopyBuffer(
            *buffer(), *dst.buffer(), 0, 0, size() * sizeof(T));
        if (err) throw OclException("Error enqueueCopyBuffer", err);
    }
};

}  // namespace core
}  // namespace fingerprint_parallel