#include "ImgStatics.hpp"

#include "ocl_core_src.hpp"

ImgStatics::ImgStatics(OclInfo oclInfo) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(ocl_src_statics);
    this->program = cl::Program(oclInfo.ctx, sources);

    cl_int err = this->program.build(oclInfo.devices);
    if (err) throw OclBuildException(err);
}

long long ImgStatics::sum(MatrixBuffer<BYTE> &src) {
    cl::Kernel kernel(program, "sum");

    int N = src.getLen();
    int group_size = 128;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    MatrixBuffer<cl_long> tmp(n_groups, 1);
    tmp.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *tmp.getClBuffer());
    kernel.setArg(2, group_size * sizeof(cl_long), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(global_work_size),
        cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);

    tmp.toHost(oclInfo);

    long long result = 0;
    long long *data = tmp.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    return result;
}

double ImgStatics::mean(MatrixBuffer<BYTE> &src) {
    const int N = src.getLen();
    double result = static_cast<double>(sum(src)) / N;
    return result;
}

double ImgStatics::var(MatrixBuffer<BYTE> &src) {
    const int N = src.getLen();
    double mean = static_cast<double>(sum(src)) / N;
    long long elementSquareSum = squareSum(src);
    double result = static_cast<double>(elementSquareSum) / N - mean * mean;
    return result;
}

long long ImgStatics::squareSum(MatrixBuffer<BYTE> &src) {
    cl::Kernel kernel(program, "squareSum");

    int N = src.getLen();
    int group_size = 128;
    int n_groups = (N + (group_size - 1)) / group_size;
    int global_work_size = group_size * n_groups;

    MatrixBuffer<cl_long> tmp(n_groups, 1);
    tmp.createBuffer(oclInfo.ctx);

    kernel.setArg(0, *src.getClBuffer());
    kernel.setArg(1, *tmp.getClBuffer());
    kernel.setArg(2, group_size * sizeof(cl_long), NULL);
    kernel.setArg(3, N);

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(global_work_size),
        cl::NDRange(group_size));

    if (err) throw OclKernelEnqueueError(err);

    tmp.toHost(oclInfo);

    long long result = 0;
    long long *data = tmp.getData();
    for (int i = 0; i < n_groups; ++i) {
        result += data[i];
    }

    return result;
}
