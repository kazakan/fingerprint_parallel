#include "ImgStatics.hpp"

void ImgStatics::sum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
    cl::Kernel kernel(program, "gray");

    kernel.setArg(0, src);
    kernel.setArg(1, dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(dst.getWidth(), dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgStatics::mean(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
}

void ImgStatics::var(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
}
