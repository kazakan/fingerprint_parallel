#include "MinutiaeDetector.hpp"

#include "ocl_core_src.hpp"

MinutiaeDetector::MinutiaeDetector(OclInfo oclInfo) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_transform);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err) throw OclBuildException(err);
}

void MinutiaeDetector::applyCrossNumber(MatrixBuffer<BYTE> &src,
                                        MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "crossNumbers");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}
