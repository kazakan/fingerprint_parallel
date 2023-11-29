#include "ImgTransform.hpp"

#include "ScalarBuffer.hpp"
#include "ocl_core_src.hpp"

namespace fingerprint_parallel {
namespace core {

ImgTransform::ImgTransform(OclInfo ocl_info) {
    this->ocl_info = ocl_info;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_transform);
    this->program = cl::Program(ocl_info.ctx_, sources);

    cl_int err = this->program.build(ocl_info.devices_);
    if (err) throw OclBuildException(err);
}

void ImgTransform::to_gray_scale(cl::Image2D &src, MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "gray");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, src);
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::negate(MatrixBuffer<uint8_t> &src,
                          MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "negate");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::normalize(MatrixBuffer<uint8_t> &src,
                             MatrixBuffer<uint8_t> &dst, float M0, float V0,
                             ScalarBuffer<float> &M, ScalarBuffer<float> &V) {
    cl::Kernel kernel(program, "normalize");

    const std::size_t group_size = 16;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, *M.buffer());
    kernel.setArg(3, *V.buffer());
    kernel.setArg(4, M0);
    kernel.setArg(5, V0);
    kernel.setArg(6, dst.width());
    kernel.setArg(7, dst.height());

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::binarize(MatrixBuffer<uint8_t> &src,
                            MatrixBuffer<uint8_t> &dst, int threshold) {
    cl::Kernel kernel(program, "binarize");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());
    kernel.setArg(4, threshold);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::dynamic_thresholding(MatrixBuffer<uint8_t> &src,
                                        MatrixBuffer<uint8_t> &dst,
                                        int block_size, float scale) {
    cl::Kernel kernel(program, "dynamicThreshold");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, src.width());
    kernel.setArg(3, src.height());
    kernel.setArg(4, block_size);
    kernel.setArg(5, scale);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

bool ImgTransform::thinning_one_iter(MatrixBuffer<uint8_t> &src,
                                     MatrixBuffer<uint8_t> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinFourCon");

    const std::size_t group_size = 16;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    MatrixBuffer<uint8_t> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.create_buffer(&ocl_info);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.buffer());  // ContinueFlags
    kernel.setArg(6, sizeof(uint8_t) * group_size * group_size,
                  nullptr);  // localContinueFlags

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);

    globalFlag.to_host();
    bool flag = false;  // whether a pixel changed
    for (int i = 0; i < globalFlag.size(); ++i) {
        flag |= globalFlag.data()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

bool ImgTransform::thinning8_one_iter(MatrixBuffer<uint8_t> &src,
                                      MatrixBuffer<uint8_t> &dst, int dir = 0) {
    cl::Kernel kernel(program, "rosenfieldThinEightCon");

    const std::size_t group_size = 16;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    MatrixBuffer<uint8_t> globalFlag(n_groups.get()[0], n_groups.get()[1]);
    globalFlag.create_buffer(&ocl_info);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());
    kernel.setArg(4, dir);
    kernel.setArg(5, *globalFlag.buffer());  // ContinueFlags
    kernel.setArg(6, sizeof(uint8_t) * group_size * group_size,
                  nullptr);  // localContinueFlags

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);

    globalFlag.to_host();
    bool flag = false;  // whether a pixel changed
    for (int i = 0; i < globalFlag.size(); ++i) {
        flag |= globalFlag.data()[i];
    }

    // if at least one pixel changed, not finished.
    return !flag;
}

void ImgTransform::thinning(MatrixBuffer<uint8_t> &src,
                            MatrixBuffer<uint8_t> &dst) {
    MatrixBuffer<uint8_t> input(src.width(), src.height());
    MatrixBuffer<uint8_t> output(dst.width(), dst.height());
    input.create_buffer(&ocl_info);
    output.create_buffer(&ocl_info);

    // copy src to input
    src.copy_buffer(input);
    int loopCnt = 0;
    const int maxLoop = 1000000;

    bool done = false;
    do {
        done = true;
        done &= thinning_one_iter(input, output, 0);
        output.copy_buffer(input);
        done &= thinning_one_iter(input, output, 1);
        output.copy_buffer(input);
        done &= thinning_one_iter(input, output, 2);
        output.copy_buffer(input);
        done &= thinning_one_iter(input, output, 3);
        output.copy_buffer(input);

        // std::cout<<"rrr"<<std::endl;

    } while (!done && (loopCnt++ < maxLoop));

    // copy output to dst
    output.copy_buffer(dst);
}

void ImgTransform::thinning8(MatrixBuffer<uint8_t> &src,
                             MatrixBuffer<uint8_t> &dst) {
    MatrixBuffer<uint8_t> input(src.width(), src.height());
    MatrixBuffer<uint8_t> output(dst.width(), dst.height());
    input.create_buffer(&ocl_info);
    output.create_buffer(&ocl_info);

    // copy src to input
    src.copy_buffer(input);
    int loopCnt = 0;
    const int maxLoop = 1000000;

    bool done = false;
    do {
        done = true;
        done &= thinning8_one_iter(input, output, 0);
        output.copy_buffer(input);
        done &= thinning8_one_iter(input, output, 1);
        output.copy_buffer(input);
        done &= thinning8_one_iter(input, output, 2);
        output.copy_buffer(input);
        done &= thinning8_one_iter(input, output, 3);
        output.copy_buffer(input);

    } while (!done && (loopCnt++ < maxLoop));
    DLOG("LOOP %d : ", loopCnt);

    // copy output to dst
    output.copy_buffer(dst);
}

void ImgTransform::gaussian_filter(MatrixBuffer<uint8_t> &src,
                                   MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "gaussian");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::copy(MatrixBuffer<uint8_t> &src,
                        MatrixBuffer<uint8_t> &dst) {
    cl::Kernel kernel(program, "copy");

    const std::size_t group_size = 512;
    const std::size_t len = std::min(src.size(), dst.size());

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((len + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, len);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

void ImgTransform::rotate(MatrixBuffer<uint8_t> &src,
                          MatrixBuffer<uint8_t> &dst, const float degree) {
    cl::Kernel kernel(program, "rotate");

    const std::size_t group_size = 8;
    const std::size_t W = dst.width();
    const std::size_t H = dst.height();

    cl::NDRange local_work_size(group_size, group_size);
    cl::NDRange n_groups((W + (group_size - 1)) / group_size,
                         (H + (group_size - 1)) / group_size);
    cl::NDRange global_work_size(group_size * n_groups.get()[0],
                                 group_size * n_groups.get()[1]);

    kernel.setArg(0, *src.buffer());
    kernel.setArg(1, *dst.buffer());
    kernel.setArg(2, dst.width());
    kernel.setArg(3, dst.height());
    kernel.setArg(4, degree);

    cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
        kernel, cl::NullRange, global_work_size, local_work_size);

    if (err) throw OclKernelEnqueueError(err);
}

}  // namespace core
}  // namespace fingerprint_parallel