
#include "main.hpp"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

string readFile(string path) {
    ifstream ifs(path);
    return string((istreambuf_iterator<char>(ifs)),
                  (istreambuf_iterator<char>()));
}

void run1() {
    string pathPrefix = "../";
    cl_int err = 0;
    FreeImage_Initialise(true);

    cout << "Running" << endl;

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    cout << "Opencl initialized" << endl;

    // init kernels
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);
    MinutiaeDetector detector(oclInfo);

    cout << "kernel loaded" << endl;

    // load image
    Img img(pathPrefix + "fingerprint.jpg");
    cout << "Loaded Image" << endl;

    // create opencl Image
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    MatrixBuffer<BYTE> buffer1(img.width, img.height);
    MatrixBuffer<BYTE> buffer2(img.width, img.height);

    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    cl::Image2D climg(oclInfo.ctx, CL_MEM_READ_WRITE, imgFormat, img.width,
                      img.height, 0, 0);

    err = oclInfo.queue.enqueueWriteImage(
        climg, CL_FALSE, {0, 0, 0}, {img.width, img.height, 1}, 0, 0, img.data);
    if (err) throw OclException("Error while enqueue image", err);

    imgTransformer.toGrayScale(climg, buffer1);
    buffer1.toHost(oclInfo);
    Img resultGray(buffer1);
    resultGray.saveImage(pathPrefix + "resultGray.png");

    // negate
    /*
    imgTransformer.negate(buffer1,buffer2);
    buffer2.toHost(oclInfo);
    Img resultNegate(buffer2);
    resultNegate.saveImage(pathPrefix + "resultNegate.png");
    */

    // normalize
    float mean = imgStatics.mean(buffer1);
    float var = imgStatics.var(buffer1);

    cout << "Mean : " << mean << " var : " << var << endl;

    imgTransformer.normalize(buffer1, buffer2, 128, 2000, mean, var);
    buffer2.toHost(oclInfo);
    Img resultNormalize(buffer2);
    resultNormalize.saveImage(pathPrefix + "resultNormalize.png");

    // gaussian filter
    imgTransformer.applyGaussianFilter(buffer2, buffer1);

    buffer1.toHost(oclInfo);
    Img resultGaussian(buffer1);
    resultGaussian.saveImage(pathPrefix + "resultGaussian.png");

    // dynamic thresholding
    // imgTransformer.applyDynamicThresholding(buffer1, buffer2, 3, 1.05);

    // negate
    imgTransformer.negate(buffer1, buffer2);

    // binarize
    imgTransformer.binarize(buffer2, buffer1);

    buffer1.toHost(oclInfo);
    Img resultThreshold(buffer1);
    resultThreshold.saveImage(pathPrefix + "resultThreshold.png");

    // thinning
    imgTransformer.applyThinning8(buffer1, buffer2);

    buffer2.toHost(oclInfo);
    Img resultThinning(buffer2);
    resultThinning.saveImage(pathPrefix + "resultThinning.png");

    // cross number
    detector.applyCrossNumber(buffer2, buffer1);
    buffer1.toHost(oclInfo);

    cout << buffer1.getLen() << endl;
    for (int i = 0; i < buffer1.getLen(); ++i) {
        BYTE val = buffer1.getData()[i];
        if (val != 0) {
            cout << "Found type " << (int)val << " at " << i << "\n";
        }
        buffer1.getData()[i] = val * 40;  // for visualize
    }

    Img resultCrossNum(buffer1);
    resultCrossNum.saveImage(pathPrefix + "resultCrossNum.png");

    FreeImage_DeInitialise();
}

void run2() {
    string pathPrefix = "../";
    cl_int err = 0;
    FreeImage_Initialise(true);

    cout << "Running" << endl;

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    cout << "Opencl initialized" << endl;

    // Load kernel
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);
    MinutiaeDetector detector(oclInfo);
    cout << "kernel loadded" << endl;

    // load image

    Img img(pathPrefix + "fingerprint.BMP");
    cout << "Loaded Image" << endl;

    // create opencl Image
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    MatrixBuffer<BYTE> buffer1(img.width, img.height);
    MatrixBuffer<BYTE> buffer2(img.width, img.height);

    buffer1.createBuffer(oclInfo.ctx);
    buffer2.createBuffer(oclInfo.ctx);

    cl::Image2D climg(oclInfo.ctx, CL_MEM_READ_WRITE, imgFormat, img.width,
                      img.height, 0, 0);

    err = oclInfo.queue.enqueueWriteImage(
        climg, CL_FALSE, {0, 0, 0}, {img.width, img.height, 1}, 0, 0, img.data);
    if (err) throw OclException("Error while enqueue image", err);

    imgTransformer.toGrayScale(climg, buffer1);
    buffer1.toHost(oclInfo);
    Img resultGray(buffer1);
    resultGray.saveImage(pathPrefix + "resultGray.png");

    // negate

    imgTransformer.negate(buffer1, buffer2);
    buffer2.toHost(oclInfo);
    Img resultNegate(buffer2);
    resultNegate.saveImage(pathPrefix + "resultNegate.png");

    // normalize
    float mean = imgStatics.mean(buffer2);
    float var = imgStatics.var(buffer2);

    cout << "Mean : " << mean << " var : " << var << endl;

    imgTransformer.normalize(buffer2, buffer1, 128, 1000, mean, var);
    buffer1.toHost(oclInfo);
    Img resultNormalize(buffer1);
    resultNormalize.saveImage(pathPrefix + "resultNormalize.png");

    // gaussian filter
    imgTransformer.applyGaussianFilter(buffer1, buffer2);

    buffer2.toHost(oclInfo);
    Img resultGaussian(buffer2);
    resultGaussian.saveImage(pathPrefix + "resultGaussian.png");

    // dynamic thresholding
    imgTransformer.applyDynamicThresholding(buffer2, buffer1, 3);

    buffer1.toHost(oclInfo);
    Img resultThreshold(buffer1);
    resultThreshold.saveImage(pathPrefix + "resultThreshold.png");

    // thinning
    imgTransformer.applyThinning(buffer1, buffer2);

    buffer2.toHost(oclInfo);
    Img resultThinning(buffer2);
    resultThinning.saveImage(pathPrefix + "resultThinning.png");

    // cross number
    detector.applyCrossNumber(buffer2, buffer1);
    buffer1.toHost(oclInfo);
    Img resultCrossNum(buffer1);
    resultCrossNum.saveImage(pathPrefix + "resultCrossNum.png");

    cout << buffer1.getLen() << endl;
    for (int i = 0; i < buffer1.getLen(); ++i) {
        BYTE val = buffer1.getData()[i];
        if (val != 0) {
            cout << "Found type " << (int)val << " at " << i << "\n";
        }
    }

    FreeImage_DeInitialise();
}

// driver code
int main(int argc, char **argv) {
    cout << argv[0] << endl;

    run1();
    // run2();
    return 0;
}