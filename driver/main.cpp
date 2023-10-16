
#include "main.hpp"

#include <memory>

#include "ImgTransform.hpp"
#include "MatrixBuffer.hpp"
#include "MinutiaeDetector.hpp"
#include "OclInfo.hpp"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

string readFile(string path) {
    ifstream ifs(path);
    return string((istreambuf_iterator<char>(ifs)),
                  (istreambuf_iterator<char>()));
}

unique_ptr<MatrixBuffer<BYTE>> preprocess(Img& img, OclInfo& oclInfo,
                                          ImgTransform& imgTransformer,
                                          ImgStatics& imgStatics,
                                          MinutiaeDetector& detector,
                                          const string& resultPrefix = "") {
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    unique_ptr<MatrixBuffer<BYTE>> mainBuffer =
        make_unique<MatrixBuffer<BYTE>>(img.width, img.height);
    MatrixBuffer<BYTE> tmpBuffer(img.width, img.height);

    cl::Image2D climg(oclInfo.ctx, CL_MEM_READ_WRITE, imgFormat, img.width,
                      img.height, 0, 0);

    int err = oclInfo.queue.enqueueWriteImage(
        climg, CL_FALSE, {0, 0, 0}, {img.width, img.height, 1}, 0, 0, img.data);
    if (err) throw OclException("Error while enqueue image", err);

    mainBuffer->createBuffer(oclInfo.ctx);
    tmpBuffer.createBuffer(oclInfo.ctx);

    imgTransformer.toGrayScale(climg, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->toHost(oclInfo);
    Img resultGray(*mainBuffer);
    resultGray.saveImage(resultPrefix + "resultGray.png");

    // negate

    imgTransformer.negate(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    mainBuffer->toHost(oclInfo);
    Img resultNegate(*mainBuffer);
    resultNegate.saveImage(resultPrefix + "resultNegate.png");

    // normalize
    float mean = imgStatics.mean(*mainBuffer);
    float var = imgStatics.var(*mainBuffer);

    imgTransformer.normalize(*mainBuffer, tmpBuffer, 128, 2000, mean, var);
    imgTransformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->toHost(oclInfo);
    Img resultNormalize(*mainBuffer);
    resultNormalize.saveImage(resultPrefix + "resultNormalize.png");

    // gaussian filter
    imgTransformer.applyGaussianFilter(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    imgTransformer.applyGaussianFilter(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    imgTransformer.applyGaussianFilter(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    imgTransformer.applyGaussianFilter(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    imgTransformer.applyGaussianFilter(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->toHost(oclInfo);
    Img resultGaussian(*mainBuffer);
    resultGaussian.saveImage(resultPrefix + "resultGaussian.png");

    // dynamic thresholding
    // imgTransformer.applyDynamicThresholding(*mainBuffer, tmpBuffer, 5, 1.00);
    // imgTransformer.copy(tmpBuffer, *mainBuffer);

    // mainBuffer->toHost(oclInfo);
    // Img resultDynamicThresholding(*mainBuffer);
    // resultDynamicThresholding.saveImage(resultPrefix+"resultDynamicThresholding.png");

    // binarize
    imgTransformer.binarize(*mainBuffer, tmpBuffer, 200);
    imgTransformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->toHost(oclInfo);
    Img resultBinarize(*mainBuffer);
    resultBinarize.saveImage(resultPrefix + "resultBinarize.png");

    // thinning
    imgTransformer.applyThinning8(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->toHost(oclInfo);
    Img resultThinning(*mainBuffer);
    resultThinning.saveImage(resultPrefix + "resultThinning.png");

    // cross number
    detector.applyCrossNumber(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    tmpBuffer.toHost(oclInfo);

    detector.removeFalseMinutiae(*mainBuffer, tmpBuffer);
    imgTransformer.copy(tmpBuffer, *mainBuffer);
    tmpBuffer.toHost(oclInfo);

    Img resultCrossNumber(tmpBuffer);

    for (int i = 0; i < tmpBuffer.getLen(); ++i) {
        BYTE val = tmpBuffer.getData()[i];
        if (val != 0) {
            // cout << "Found type " << (int)val << " at " << i << "\n";
            tmpBuffer.getData()[i] = 255;  // for visualize

            if (val == 1) {  // B
                resultCrossNumber.data[i * 4] = 255;
                resultCrossNumber.data[i * 4 + 1] = 0;
                resultCrossNumber.data[i * 4 + 2] = 0;
                resultCrossNumber.data[i * 4 + 3] = 255;
            } else if (val == 3) {  // G
                resultCrossNumber.data[i * 4] = 0;
                resultCrossNumber.data[i * 4 + 1] = 255;
                resultCrossNumber.data[i * 4 + 2] = 0;
                resultCrossNumber.data[i * 4 + 3] = 255;
            } else if (val == 4) {  // R
                resultCrossNumber.data[i * 4] = 0;
                resultCrossNumber.data[i * 4 + 1] = 0;
                resultCrossNumber.data[i * 4 + 2] = 255;
                resultCrossNumber.data[i * 4 + 3] = 255;
            }

        } else {
            tmpBuffer.getData()[i] = 0;
        }
    }

    resultCrossNumber.saveImage(resultPrefix + "resultCrossNumber.png");

    return mainBuffer;
}

void run1() {
    string pathPrefix = "./data/DB1_B/";
    cl_int err = 0;
    FreeImage_Initialise(true);

    // Show opencl information
    OclInfo::showPlatformInfos();

    LOG("Running");

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    DLOG("Opencl initialized");

    // init kernels
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);
    MinutiaeDetector detector(oclInfo);

    LOG("kernel loaded");

    // load image
    Img img1(pathPrefix + "101_2.tif");
    Img img2(pathPrefix + "101_4.tif");

    LOG("Image loaded");

    unique_ptr<MatrixBuffer<BYTE>> buffer1 = preprocess(
        img1, oclInfo, imgTransformer, imgStatics, detector, "img1_");
    unique_ptr<MatrixBuffer<BYTE>> buffer2 = preprocess(
        img2, oclInfo, imgTransformer, imgStatics, detector, "img2_");

    LOG("Image Preprocessed");

    FreeImage_DeInitialise();
}

void identicalRun() {
    string pathPrefix = "./data/DB1_B/";
    cl_int err = 0;
    FreeImage_Initialise(true);

    // Show opencl information
    OclInfo::showPlatformInfos();

    LOG("Running");

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    DLOG("Opencl initialized");

    // init kernels
    ImgTransform imgTransformer(oclInfo);
    ImgStatics imgStatics(oclInfo);
    MinutiaeDetector detector(oclInfo);

    LOG("kernel loaded");

    // load image
    Img img1(pathPrefix + "101_2.tif");
    Img img2(pathPrefix + "101_2.tif");

    LOG("Image loaded");

    unique_ptr<MatrixBuffer<BYTE>> buffer1 = preprocess(
        img1, oclInfo, imgTransformer, imgStatics, detector, "img1_");
    unique_ptr<MatrixBuffer<BYTE>> buffer2 = preprocess(
        img2, oclInfo, imgTransformer, imgStatics, detector, "img2_");

    LOG("Image Preprocessed");

    FreeImage_DeInitialise();
}

// driver code
int main(int argc, char** argv) {
    cout << argv[0] << endl;

    identicalRun();
    // run1();
    return 0;
}