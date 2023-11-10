#include "ImgStatics.hpp"

#include <CL/cl_platform.h>

#include <cstdint>

#include "ScalarBuffer.hpp"
#include "ocl_core_src.hpp"

namespace fingerprint_parallel {
namespace core {

ImgStatics::ImgStatics(OclInfo ocl_info) {
    this->ocl_info = ocl_info;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_statics);
    this->program = cl::Program(ocl_info.ctx_, sources);

    cl_int err = this->program.build(ocl_info.devices_);
    if (err) throw OclBuildException(err);
}

void ImgStatics::sum(MatrixBuffer<uint8_t> &src, ScalarBuffer<uint64_t> &ret) {
    cl::Kernel kernel(program, "sum");

    const int N = src.size();
    const int group_size = 512;

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *ret.buffer());
    kernel.setArg(2, group_size * sizeof(int64_t), NULL);
    kernel.setArg(3, N);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                      cl::NDRange(group_size),
                                                      cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);
}

void ImgStatics::square_sum(MatrixBuffer<uint8_t> &src,
                            ScalarBuffer<uint64_t> &ret) {
    cl::Kernel kernel(program, "squareSum");

    const int N = src.size();
    const int group_size = 512;

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *ret.buffer());
    kernel.setArg(2, group_size * sizeof(uint64_t), NULL);
    kernel.setArg(3, N);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                      cl::NDRange(group_size),
                                                      cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);
}

void ImgStatics::mean(MatrixBuffer<uint8_t> &src, ScalarBuffer<cl_float> &ret) {
    cl::Kernel kernel(program, "mean");

    const int N = src.size();
    const int group_size = 512;

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *ret.buffer());
    kernel.setArg(2, group_size * sizeof(cl_long), NULL);
    kernel.setArg(3, N);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                      cl::NDRange(group_size),
                                                      cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);
}

void ImgStatics::var(MatrixBuffer<uint8_t> &src, ScalarBuffer<cl_float> &ret) {
    cl::Kernel kernel(program, "var");

    const int N = src.size();
    const int group_size = 512;

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *ret.buffer());
    kernel.setArg(2, group_size * sizeof(uint64_t), NULL);
    kernel.setArg(3, group_size * sizeof(uint64_t), NULL);
    kernel.setArg(4, N);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                      cl::NDRange(group_size),
                                                      cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);
}

}  // namespace core
}  // namespace fingerprint_parallel