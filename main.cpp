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

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    cout<<"Opencl initialized"<<endl;

    // Load kernel
    string transformSource = readFile(pathPrefix + "transform.cl");
    string staticsSoure = readFile(pathPrefix + "statics.cl");
    cout << "kernel file loadded" << endl;

    ImgTransform imgTransformer(oclInfo,transformSource);

    // create opencl Image
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    MatrixBuffer<BYTE> buffer1(img.width, img.height);
    MatrixBuffer<BYTE> buffer2(img.width, img.height);

    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    cl::Image2D climg(
        oclInfo.ctx,
        CL_MEM_READ_WRITE,
        imgFormat,
        img.width,
        img.height,
        0,
        0);

    err = oclInfo.queue.enqueueWriteImage(climg, CL_FALSE, {0, 0, 0}, {img.width, img.height, 1}, 0, 0, img.data);
    if(err) throw OclException("Error while enqueue image", err);

    imgTransformer.toGrayScale(climg,buffer1);
    imgTransformer.normalize(buffer2,buffer1,128,5,128,5);
    imgTransformer.applyGaussianFilter(buffer2,buffer1);
    imgTransformer.applyDynamicThresholding(buffer1,buffer2,9);

    // ger return value
    err = oclInfo.queue.enqueueReadBuffer(*buffer2.getClBuffer(), CL_TRUE, 0, buffer2.getLen(), buffer2.getData(), nullptr, nullptr);
    if(err) throw OclException("Error enqueueReadBuffer", err);

    // write image
    Img resultImg2(buffer2);
    bool saved = resultImg2.saveImage(pathPrefix + "result222.png");
    if (!saved) {
        cerr << "Failed save image" << endl;
    }
    FreeImage_DeInitialise();

    // delete [] outImgData;

    return 0;
}