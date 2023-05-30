#include "MinutiaeDetector.hpp"

MinutiaeDetector::MinutiaeDetector(OclInfo oclInfo, string source) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(source);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err)
        throw OclBuildException(err);
}

void MinutiaeDetector::applyCrossNumber(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "crossNumbers");

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(dst.getWidth(), dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}
