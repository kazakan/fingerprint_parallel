#include "ImgTransform.hpp"

#include "ocl_core_src.hpp"

ImgTransform::ImgTransform(OclInfo oclInfo) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_transform);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err) throw OclBuildException(err);
}

void ImgTransform::toGrayScale(cl::Image2D &src, MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "gray");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    kernel.setArg(0, src);
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::negate(MatrixBuffer<uint8_t> &src,
                          MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "negate");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

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

void ImgTransform::normalize(MatrixBuffer<uint8_t> &src,
                             MatrixBuffer<uint8_t> &dst, float M0, float V0,
                             float M, float V) {
    cl::Kernel kernel(program, "normalize");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, M);
    kernel.setArg(3, V);
    kernel.setArg(4, M0);
    kernel.setArg(5, V0);
    kernel.setArg(6, dst.getWidth());
    kernel.setArg(7, dst.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::binarize(MatrixBuffer<uint8_t> &src,
                            MatrixBuffer<uint8_t> &dst, int threshold) {
    cl::Kernel kernel(program, "binarize");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, threshold);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::applyDynamicThresholding(MatrixBuffer<uint8_t> &src,
                                            MatrixBuffer<uint8_t> &dst,
                                            int blockSize, float scale) {
    cl::Kernel kernel(program, "dynamicThreshold");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, src.getWidth());
    kernel.setArg(3, src.getHeight());
    kernel.setArg(4, blockSize);
    kernel.setArg(5, scale);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

bool ImgTransform::thinningOneIter(MatrixBuffer<uint8_t> &src,
                                   MatrixBuffer<uint8_t> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinFourCon");

    const std::size_t groupSize = 16;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    MatrixBuffer<uint8_t> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.getClBuffer());  // ContinueFlags
    kernel.setArg(6, sizeof(uint8_t) * groupSize * groupSize,
                  nullptr);  // localContinueFlags

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);

    globalFlag.toHost(oclInfo);
    bool flag = false;  // whether a pixel changed
    for (int i = 0; i < globalFlag.getLen(); ++i) {
        flag |= globalFlag.getData()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

bool ImgTransform::thinning8OneIter(MatrixBuffer<uint8_t> &src,
                                    MatrixBuffer<uint8_t> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinEightCon");

    const std::size_t groupSize = 16;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((W + (groupSize - 1)) / groupSize,
                         (H + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0],
                                 groupSize * n_groups.get()[1]);

    MatrixBuffer<uint8_t> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, dst.getWidth());
    kernel.setArg(3, dst.getHeight());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.getClBuffer());  // ContinueFlags
    kernel.setArg(6, sizeof(uint8_t) * groupSize * groupSize,
                  nullptr);  // localContinueFlags

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);

    globalFlag.toHost(oclInfo);
    bool flag = false;  // whether a pixel changed
    for (int i = 0; i < globalFlag.getLen(); ++i) {
        flag |= globalFlag.getData()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

void ImgTransform::applyThinning(MatrixBuffer<uint8_t> &src,
                                 MatrixBuffer<uint8_t> &dst) {
    MatrixBuffer<uint8_t> input(src.getWidth(), src.getHeight());
    MatrixBuffer<uint8_t> output(dst.getWidth(), dst.getHeight());
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

void ImgTransform::applyThinning8(MatrixBuffer<uint8_t> &src,
                                  MatrixBuffer<uint8_t> &dst) {
    MatrixBuffer<uint8_t> input(src.getWidth(), src.getHeight());
    MatrixBuffer<uint8_t> output(dst.getWidth(), dst.getHeight());
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

void ImgTransform::applyGaussianFilter(MatrixBuffer<uint8_t> &src,
                                       MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "gaussian");

    const std::size_t groupSize = 8;
    const std::size_t W = dst.getWidth();
    const std::size_t H = dst.getHeight();

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

void ImgTransform::copy(MatrixBuffer<uint8_t> &src,
                        MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "copy");

    const std::size_t groupSize = 512;
    const std::size_t len = std::min(src.getLen(), dst.getLen());

    cl::NDRange local_work_size(groupSize, groupSize);
    cl::NDRange n_groups((len + (groupSize - 1)) / groupSize);
    cl::NDRange global_work_size(groupSize * n_groups.get()[0]);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, len);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}
