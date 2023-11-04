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

/**
 * @brief Class represents Matrix. May contains related Opencl Buffer object.
 *
 */
template <typename T>
class MatrixBuffer {
   private:
    T *_data = nullptr;
    std::size_t _width;
    std::size_t _height;
    std::size_t _len;
    cl::Buffer *_buffer = nullptr;

   public:
    /**
     * @brief Create MatruxBuffer with width, and height.
     * @param width width of image. Number of pixels in row.
     * @param height height of image. Number of pixels in column.
     */
    MatrixBuffer(std::size_t width, std::size_t height)
        : _width(width), _height(height), _len(width * height) {
        _data = new T[_len];
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
        : _width(width), _height(height), _len(width * height) {
        _data = new T[_len];

        const std::size_t limit =
            std::min(_len, static_cast<std::size_t>(data.size()));
        for (int i = 0; i < limit; ++i) {
            _data[i] = data[i];
        }
    };

    ~MatrixBuffer() {
        delete[] _data;
        _data = nullptr;
        if (_buffer != nullptr) delete _buffer;
    }

    bool operator==(const MatrixBuffer<T> &rhs) const {
        if (_width != rhs._width || _height != rhs._height || _len != rhs._len)
            return false;

        for (int i = 0; i < _len; ++i) {
            if (_data[i] != rhs._data[i]) return false;
        }
        return true;
    }

    /**
     * @brief Initialize OpenCL Buffer for matrix.
     * @param ctx cl::Context object
     * @param memFlag flag used for OpenCL memory access policy. Default =
     * CL_MEM_READ_WRITE
     */
    void createBuffer(cl::Context ctx,
                      cl_mem_flags memFlag = CL_MEM_READ_WRITE) {
        cl_int err = CL_SUCCESS;
        _buffer = new cl::Buffer(ctx, memFlag, _len * sizeof(T), nullptr, &err);

        if (err != CL_SUCCESS) {
            throw OclException(
                "Error while creating OCL buffer in MatrixBuffer.", err);
        }
    }

    /**
     * @brief Get width of matrix.
     * @return Width of matrix
     */
    const std::size_t getWidth() const { return _width; }

    /**
     * @brief Get height of matrix.
     * @return Height of matrix
     */
    const std::size_t getHeight() const { return _height; }

    /**
     * @brief Get numbers of elements in matrix. (width*height)
     * @return width*height
     */
    const std::size_t getLen() const { return _len; }

    /**
     * @brief  Get pointer that points first element of matrix.
     * @return Pointer to first element.
     */
    T *getData() { return _data; }

    /**
     * @brief Get Related OpenCL Buffer.
     * @return Related OpenCl Buffer
     */
    cl::Buffer *getClBuffer() { return _buffer; }

    /**
     * @brief Copy Host memory to Gpu.
     * @param oclInfo OclInfo contains valid queue.
     * @param blocking if false, only enqueue job and continue. Default=true.
     */
    void toGpu(OclInfo &oclInfo, bool blocking = true) {
        cl_int err = oclInfo.queue.enqueueWriteBuffer(
            *getClBuffer(), blocking, 0, getLen() * sizeof(T),
            (void *)getData(), nullptr, nullptr);
        if (err) throw OclException("Error enqueueWriteBuffer", err);
    }

    /**
     * @brief Copy Gpu memory to host.
     * @param oclInfo OclInfo contains valid queue.
     * @param blocking if false, only enqueue job and continue. Default=true.
     */
    void toHost(OclInfo &oclInfo, bool blocking = true) {
        cl_int err = oclInfo.queue.enqueueReadBuffer(
            *getClBuffer(), blocking, 0, getLen() * sizeof(T),
            (void *)getData(), nullptr, nullptr);
        if (err) throw OclException("Error enqueueReadBuffer", err);
    }

    /**
     * @brief Use enqueueCopyBuffer to copy buffer in gpu.
     * @param oclInfo OclInfo contains valid queue.
     * @param dst destination to be copied.
     */
    void copyBuffer(OclInfo &oclInfo, MatrixBuffer &dst) {
        cl_int err = oclInfo.queue.enqueueCopyBuffer(
            *getClBuffer(), *dst.getClBuffer(), 0, 0, getLen() * sizeof(T));
        if (err) throw OclException("Error enqueueCopyBuffer", err);
    }
};
