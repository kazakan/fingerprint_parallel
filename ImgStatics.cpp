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

float ImgStatics::sum(MatrixBuffer<BYTE> &src) {
    cl::Kernel kernel(program, "sum");

    int N = src.getLen();
    int group_size = 128;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    MatrixBuffer<cl_float> tmp(n_groups, 1);
    tmp.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *tmp.getClBuffer());
    kernel.setArg(2, group_size * sizeof(cl_float), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NDRange(group_size));

    if (err)
        throw OclKernelEnqueueError(err);

    tmp.toHost(oclInfo);

    float result = 0;
    float *data = tmp.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    return result;
}

float ImgStatics::mean(MatrixBuffer<BYTE> &src) {
    const int N = src.getLen();
    float result = sum(src) / N;
    return result;
}

float ImgStatics::var(MatrixBuffer<BYTE> &src) {
    const int N = src.getLen();
    float mean = sum(src) / N;
    float elementSquareSum = squareSum(src);
    float result = elementSquareSum / N - mean * mean;
    return result;
}

float ImgStatics::squareSum(MatrixBuffer<BYTE> &src) {
    cl::Kernel kernel(program, "squareSum");

    int N = src.getLen();
    int group_size = 128;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    MatrixBuffer<float> tmp(n_groups, 1);
    tmp.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *tmp.getClBuffer());
    kernel.setArg(2, group_size * sizeof(float), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NDRange(group_size));

    if (err)
        throw OclKernelEnqueueError(err);

    tmp.toHost(oclInfo);

    float result = 0;
    float *data = tmp.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    return result;
}
