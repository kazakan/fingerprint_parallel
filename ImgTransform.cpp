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
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    kernel.setArg(0, src);
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

void ImgTransform::negate(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "negate");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
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
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

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
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, src.getWidth());
    kernel.setArg(3, src.getHeight());
    kernel.setArg(4, blockSize);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}

bool ImgTransform::thinningOneIter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinFourCon");

    const size_t groupSize = 16;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    MatrixBuffer<BYTE> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.getClBuffer());                     // ContinueFlags
    kernel.setArg(6, sizeof(BYTE) * groupSize * groupSize, nullptr); // localContinueFlags

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);

    globalFlag.toHost(oclInfo);
    bool flag = false; // whether a pixel changed
    for (int i = 0; i < globalFlag.getLen(); ++i) {
        flag |= globalFlag.getData()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

bool ImgTransform::thinning8OneIter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinEightCon");

    const size_t groupSize = 16;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    MatrixBuffer<BYTE> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.getClBuffer());                     // ContinueFlags
    kernel.setArg(6, sizeof(BYTE) * groupSize * groupSize, nullptr); // localContinueFlags

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);

    globalFlag.toHost(oclInfo);
    bool flag = false; // whether a pixel changed
    for (int i = 0; i < globalFlag.getLen(); ++i) {
        flag |= globalFlag.getData()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

void ImgTransform::applyThinning(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    MatrixBuffer<BYTE> input(src.getWidth(), src.getHeight());
    MatrixBuffer<BYTE> output(dst.getWidth(), dst.getHeight());
    input.createBuffer(oclInfo.ctx);
    output.createBuffer(oclInfo.ctx);

    // copy src to input
    src.copyBuffer(oclInfo, input);
    int loopCnt = 0;
    const int maxLoop = 1000000;

    bool done = false;
    do {
        done = true;
        done &= thinningOneIter(input, output, 0);
        output.copyBuffer(oclInfo, input);
        done &= thinningOneIter(input, output, 1);
        output.copyBuffer(oclInfo, input);
        done &= thinningOneIter(input, output, 2);
        output.copyBuffer(oclInfo, input);
        done &= thinningOneIter(input, output, 3);
        output.copyBuffer(oclInfo, input);

        // std::cout<<"rrr"<<std::endl;

    } while (!done && (loopCnt++ < maxLoop));

    // copy output to dst
    output.copyBuffer(oclInfo, dst);
}

void ImgTransform::applyThinning8(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    MatrixBuffer<BYTE> input(src.getWidth(), src.getHeight());
    MatrixBuffer<BYTE> output(dst.getWidth(), dst.getHeight());
    input.createBuffer(oclInfo.ctx);
    output.createBuffer(oclInfo.ctx);

    // copy src to input
    src.copyBuffer(oclInfo, input);
    int loopCnt = 0;
    const int maxLoop = 1000000;

    bool done = false;
    do {

        done = true;
        done &= thinning8OneIter(input, output, 0);
        output.copyBuffer(oclInfo, input);
        done &= thinning8OneIter(input, output, 1);
        output.copyBuffer(oclInfo, input);
        done &= thinning8OneIter(input, output, 2);
        output.copyBuffer(oclInfo, input);
        done &= thinning8OneIter(input, output, 3);
        output.copyBuffer(oclInfo, input);

    } while (!done && (loopCnt++ < maxLoop));

    // copy output to dst
    output.copyBuffer(oclInfo, dst);
}

void ImgTransform::applyGaussianFilter(MatrixBuffer<BYTE> &src, MatrixBuffer<BYTE> &dst) {
    cl::Kernel kernel(program, "gaussian");

    const size_t groupSize = 8;
    const int W = dst.getWidth();
    const int H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize, (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0], groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

    if (err)
        throw OclKernelEnqueueError(err);
}
