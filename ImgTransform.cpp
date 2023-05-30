#include "ImgTransform.hpp"

ImgTransform::ImgTransform(OclInfo oclInfo, string source) {
    this->oclInfo = oclInfo;
    cl::Program::Sources sources;
    sources.push_back(source);
    this->program = cl::Program(oclInfo.ctx,sources);

    cl_int err = this->program.build(oclInfo.devices);
    if(err) throw OclBuildException(err);
}

void ImgTransform::toGrayScale(Img &src, MatrixBuffer &dst) {
    cl::Kernel kernel(program,"gray");
}

void ImgTransform::normalize(MatrixBuffer &src, MatrixBuffer &dst) {
}

void ImgTransform::applyGaborFilter(MatrixBuffer &src, MatrixBuffer &dst) {
}

void ImgTransform::applyDynamicThresholding(MatrixBuffer &src, MatrixBuffer &dst) {
    cl::Kernel kernel(program,"dynamicThreshold");

    kernel.setArg(0, src.getClBuffer());
    kernel.setArg(1, dst.getClBuffer());
    kernel.setArg(2, src.getWidth());
    kernel.setArg(3, src.getHeight());
    kernel.setArg(4, 9);

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(src.getWidth(), src.getHeight());

    cl_int err = oclInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
    
    if(err) throw OclBuildException(err);
}

void ImgTransform::applyThinning(MatrixBuffer &src, MatrixBuffer &dst) {
}

void ImgTransform::applyGaussianFilter(MatrixBuffer &src, MatrixBuffer &dst) {
}
