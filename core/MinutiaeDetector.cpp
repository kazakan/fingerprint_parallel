#include "MinutiaeDetector.hpp"

#include "ocl_core_src.hpp"

namespace fingerprint_parallel {
namespace core {

MinutiaeDetector::MinutiaeDetector(OclInfo ocl_info) {
    this->ocl_info_ = ocl_info;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_transform);
    this->program_ = cl::Program(ocl_info.ctx_, sources);

    cl_int err = this->program_.build(ocl_info.devices_);
    if (err) throw OclBuildException(err);
}

void MinutiaeDetector::apply_cross_number(MatrixBuffer<uint8_t> &src,
                                          MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program_, "crossNumbers");

    const size_t group_size = 8;
    const int W = dst.width();
    const int H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());

    cl_int err = ocl_info_.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void MinutiaeDetector::remove_false_minutiae(MatrixBuffer<uint8_t> &src,
                                             MatrixBuffer<uint8_t> &dst) {
    // currently only removes points with cn=2
    cl::Kernel kernel(program_, "removeFalseMinutiae");

    const size_t group_size = 512;
    const cl_int len = std::min(src.size(), dst.size());

    cl::NDRange local_work_size(group_size);
    cl::NDRange n_groups((len + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, len);

    cl_int err = ocl_info_.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

}  // namespace core
}  // namespace fingerprint_parallel