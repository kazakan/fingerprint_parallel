#include "ImgStatics.hpp"

ImgStatics::ImgStatics(OclInfo oclInfo, string source) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(source);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err)
        throw OclBuildException(err);
}

float ImgStatics::sum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
    cl::Kernel kernel(program, "sum");

    int N = src.getLen();
    int group_size = 64;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, group_size * sizeof(float), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, 0, global_work_size, group_size);

    if (err)
        throw OclKernelEnqueueError(err);

    dst.toHost(oclInfo);

    float result = 0;
    float *data = dst.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    // copy sum of all workgroups in first of dst buffer element.
    data[0] = result;
    dst.toGpu(oclInfo);
    return result;
}

float ImgStatics::mean(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
    const int N = src.getLen();
    float result = sum(src,dst) / N;
    dst.getData()[0] = result;
    dst.toGpu(oclInfo);
    return result;
}

float ImgStatics::var(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
    const int N = src.getLen();
    float mean = sum(src,dst) / N;
    float elementSquareSum = squareSum(src,dst);
    float result = elementSquareSum / N - mean*mean;
    dst.getData()[0] = result;
    dst.toGpu(oclInfo);
    return result;
}

float ImgStatics::squareSum(MatrixBuffer<BYTE> &src, MatrixBuffer<float> &dst) {
    cl::Kernel kernel(program, "squareSum");

    int N = src.getLen();
    int group_size = 64;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *dst.getClBuffer());
    kernel.setArg(2, group_size * sizeof(float), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, 0, global_work_size, group_size);

    if (err)
        throw OclKernelEnqueueError(err);

    dst.toHost(oclInfo);

    float result = 0;
    float *data = dst.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    // copy sum of all workgroups in first of dst buffer element.
    data[0] = result;
    dst.toGpu(oclInfo);
    return result;
}
