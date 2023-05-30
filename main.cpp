// CmakeOpenclManualLink.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include "main.hpp"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

string readFile(string path) {
    ifstream ifs(path);
    return string((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));
}

// driver code
int main(int argc, char **argv) {
    string pathPrefix = "../";
    cl_int err = 0;
    FreeImage_Initialise(true);

    cout << argv[0] << endl;

    cout << "Running" << endl;
    // load image

    Img img(pathPrefix + "icon.png");
    cout << "Loaded Image" << endl;

    // Load kernel

    string sourceStr = readFile(pathPrefix + "transform.cl");
    cout << "kernel loadded" << endl;

    // init opencl

    OclInfo oclinfo = OclInfo::initOpenCL();

    // create opencl Image
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    MatrixBuffer<BYTE> buffer1(img.width,img.height);
    MatrixBuffer<BYTE> buffer2(img.width,img.height);

    buffer1.createBuffer(oclinfo.ctx);
    buffer2.createBuffer(oclinfo.ctx);

    cl::Image2D climg(
        oclinfo.ctx,
        CL_MEM_READ_WRITE,
        imgFormat,
        img.width,
        img.height,
        0,
        0);

    err = oclinfo.queue.enqueueWriteImage(climg, CL_FALSE, {0, 0, 0}, {img.width, img.height, 1}, 0, 0, img.data);
    throw OclException("Error while enqueue image",err);

    // create kernel from source code
    cl::Program::Sources sources;
    sources.push_back(sourceStr);
    cl::Program program(oclinfo.ctx, sources);
    err = program.build(oclinfo.devices);
    throw OclBuildException(err);

    cl::Kernel kernel(program, "gray");
    cl::Kernel dynamicThresholdKernel(program, "dynamicThreshold");

    // set kernel arg
    kernel.setArg(0, climg);
    kernel.setArg(1, *buffer1.getClBuffer());
    kernel.setArg(2, img.width);
    kernel.setArg(3, img.height);

    // dynamic thresholding
    dynamicThresholdKernel.setArg(0, *buffer1.getClBuffer());
    dynamicThresholdKernel.setArg(1, *buffer2.getClBuffer());
    dynamicThresholdKernel.setArg(2, img.width);
    dynamicThresholdKernel.setArg(3, img.height);
    dynamicThresholdKernel.setArg(4, 9);

    const size_t wsize = 8;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(img.width, img.height);

    // launch kernel
    err = oclinfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
    throw OclKernelEnqueueError(err);

    // ger return value
    err = oclinfo.queue.enqueueReadBuffer(*buffer1.getClBuffer(), CL_TRUE,0,buffer1.getLen(),buffer1.getData(),nullptr,nullptr);
    throw OclException("Error enqueueReadBuffer",err);
    Img resultImg1(buffer1);
    bool saved = resultImg1.saveImage(pathPrefix+"result1.png");
    if(!saved){
        cerr<<"Failed save image"<<endl;
    }

    // run kernel 
    err = oclinfo.queue.enqueueNDRangeKernel(dynamicThresholdKernel, cl::NullRange, global_work_size, local_work_size);
    throw OclKernelEnqueueError(err);

    // ger return value
    err = oclinfo.queue.enqueueReadBuffer(*buffer2.getClBuffer(), CL_TRUE,0,buffer2.getLen(),buffer2.getData(),nullptr,nullptr);
    throw OclException("Error enqueueReadBuffer",err);

    // write image
    Img resultImg2(buffer2);
    saved = resultImg2.saveImage(pathPrefix+"result2.png");
    if(!saved){
        cerr<<"Failed save image"<<endl;
    }
    FreeImage_DeInitialise();

    // delete [] outImgData;

    return 0;
}