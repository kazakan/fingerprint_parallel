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

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize -1))/groupSize,(H + (groupSize -1))/groupSize);
    cl::NDRange global_work_size(groupSize*n_groups.get()[0], groupSize*n_groups.get()[1]);

    kernel.setArg(0, src);
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::normalize(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, float M0, float V0, float M, float V) {
    cl::Kernel kernel(program, "normalize");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize -1))/groupSize,(H + (groupSize -1))/groupSize);
    cl::NDRange global_work_size(groupSize*n_groups.get()[0], groupSize*n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, M);
    kernel.setArg(3, V);
    kernel.setArg(4, M0);
    kernel.setArg(5, V0);
    kernel.setArg(6, dst.getWidth());
    kernel.setArg(7, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::applyDynamicThresholding(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int blockSize) {
    cl::Kernel kernel(program, "dynamicThreshold");

   const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize -1))/groupSize,(H + (groupSize -1))/groupSize);
    cl::NDRange global_work_size(groupSize*n_groups.get()[0], groupSize*n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, src.getWidth());
    kernel.setArg(3, src.getHeight());
    kernel.setArg(4, blockSize);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::applyThinning(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "rosenfieldThinFourCon");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize -1))/groupSize,(H + (groupSize -1))/groupSize);
    cl::NDRange global_work_size(groupSize*n_groups.get()[0], groupSize*n_groups.get()[1]);

    MatrixBuffer<BYTE> globalFlag(8, 8);
    globalFlag.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, *globalFlag.getClBuffer());             // localContinueFlags
    kernel.setArg(5, sizeof(BYTE) * groupSize * groupSize, nullptr); // localContinueFlags

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::applyGaussianFilter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "gaussian");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize -1))/groupSize,(H + (groupSize -1))/groupSize);
    cl::NDRange global_work_size(groupSize*n_groups.get()[0], groupSize*n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}
