#include "ImgTransform.hpp"

ImgTransform::ImgTransform(OclInfo oclInfo, string source) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(source);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err)
        throw OclBuildException(err);
}

void ImgTransform::toGrayScale(cl::Image2D &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "gray");

    kernel.setArg(0, src);
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

void ImgTransform::normalize(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, float M0, float V0, float M, float V) {
    cl::Kernel kernel(program, "normalize");

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, M);
    kernel.setArg(3, V);
    kernel.setArg(4, M0);
    kernel.setArg(5, V0);
    kernel.setArg(6, dst.getWidth());
    kernel.setArg(6, dst.getHeight());

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(dst.getWidth(), dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::applyDynamicThresholding(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int blockSize) {
    cl::Kernel kernel(program, "dynamicThreshold");

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(src.getWidth(), src.getHeight());

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, src.getWidth());
    kernel.setArg(3, src.getHeight());
    kernel.setArg(4, sizeof(BYTE) * wsize * wsize, nullptr); // localContinueFlags
    kernel.setArg(5, 9);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::applyThinning(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "normalize");

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

void ImgTransform::applyGaussianFilter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "gaussian");

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
