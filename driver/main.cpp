
#include "main.hpp"

#include <FreeImage.h>

#include <memory>

#include "ImgTransform.hpp"
#include "MatrixBuffer.hpp"
#include "MinutiaeDetector.hpp"
#include "OclInfo.hpp"
#include "ScalarBuffer.hpp"

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

namespace fingerprint_parallel {
namespace driver {

using namespace fingerprint_parallel::core;

string readFile(string path) {
    ifstream ifs(path);
    return string((istreambuf_iterator<char>(ifs)),
                  (istreambuf_iterator<char>()));
}

unique_ptr<MatrixBuffer<BYTE>> preprocess(Img& img, OclInfo& ocl_info,
                                          ImgTransform& img_transformer,
                                          ImgStatics& img_statics,
                                          MinutiaeDetector& detector,
                                          const string& resultPrefix = "") {
    cl::ImageFormat imgFormat(CL_RGBA, CL_UNSIGNED_INT8);
    unique_ptr<MatrixBuffer<BYTE>> mainBuffer =
        make_unique<MatrixBuffer<BYTE>>(img.width(), img.height());
    MatrixBuffer<BYTE> tmpBuffer(img.width(), img.height());

    cl::Image2D climg(ocl_info.ctx_, CL_MEM_READ_WRITE, imgFormat, img.width(),
                      img.height(), 0, 0);

    int err = ocl_info.queue_.enqueueWriteImage(climg, CL_FALSE, {0, 0, 0},
                                                {img.width(), img.height(), 1},
                                                0, 0, img.data());
    if (err) throw OclException("Error while enqueue image", err);

    mainBuffer->create_buffer(&ocl_info);
    tmpBuffer.create_buffer(&ocl_info);

    img_transformer.to_gray_scale(climg, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->to_host();
    Img resultGray(*mainBuffer);
    resultGray.save_image(resultPrefix + "resultGray.png");

    // negate

    img_transformer.negate(*mainBuffer, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);
    mainBuffer->to_host();
    Img resultNegate(*mainBuffer);
    resultNegate.save_image(resultPrefix + "resultNegate.png");

    // gaussian filter
    img_transformer.gaussian_filter(*mainBuffer, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->to_host();
    Img resultGaussian(*mainBuffer);
    resultGaussian.save_image(resultPrefix + "resultGaussian.png");

    // normalize
    ScalarBuffer<float> mean;
    ScalarBuffer<float> var;

    mean.create_buffer(&ocl_info);
    var.create_buffer(&ocl_info);

    img_statics.mean(*mainBuffer, mean);
    img_statics.var(*mainBuffer, var);

    img_transformer.normalize(*mainBuffer, tmpBuffer, 128, 1000, mean, var);
    img_transformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->to_host();
    Img resultNormalize(*mainBuffer);
    resultNormalize.save_image(resultPrefix + "resultNormalize.png");

    // dynamic thresholding
    // img_transformer.applyDynamicThresholding(*mainBuffer, tmpBuffer,
    // 3, 1.001); img_transformer.copy(tmpBuffer, *mainBuffer);

    // mainBuffer->toHost(ocl_info);
    // Img resultDynamicThresholding(*mainBuffer);
    // resultDynamicThresholding.saveImage(resultPrefix+"resultDynamicThresholding.png");

    // binarize
    img_transformer.binarize(*mainBuffer, tmpBuffer, 200);
    img_transformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->to_host();
    Img resultBinarize(*mainBuffer);
    resultBinarize.save_image(resultPrefix + "resultBinarize.png");

    // thinning
    img_transformer.thinning8(*mainBuffer, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);

    mainBuffer->to_host();
    Img resultThinning(*mainBuffer);
    resultThinning.save_image(resultPrefix + "resultThinning.png");

    // cross number
    detector.apply_cross_number(*mainBuffer, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);
    tmpBuffer.to_host();

    detector.remove_false_minutiae(*mainBuffer, tmpBuffer);
    img_transformer.copy(tmpBuffer, *mainBuffer);
    tmpBuffer.to_host();

    Img resultCrossNumber(tmpBuffer);

    for (int i = 0; i < tmpBuffer.size(); ++i) {
        BYTE val = tmpBuffer.data()[i];
        if (val != 0) {
            // cout << "Found type " << (int)val << " at " << i << "\n";
            tmpBuffer.data()[i] = 255;  // for visualize

            if (val == 1) {  // B
                resultCrossNumber.data()[i * 4] = 255;
                resultCrossNumber.data()[i * 4 + 1] = 0;
                resultCrossNumber.data()[i * 4 + 2] = 0;
                resultCrossNumber.data()[i * 4 + 3] = 255;
            } else if (val == 3) {  // G
                resultCrossNumber.data()[i * 4] = 0;
                resultCrossNumber.data()[i * 4 + 1] = 255;
                resultCrossNumber.data()[i * 4 + 2] = 0;
                resultCrossNumber.data()[i * 4 + 3] = 255;
            } else if (val == 4) {  // R
                resultCrossNumber.data()[i * 4] = 0;
                resultCrossNumber.data()[i * 4 + 1] = 0;
                resultCrossNumber.data()[i * 4 + 2] = 255;
                resultCrossNumber.data()[i * 4 + 3] = 255;
            }

        } else {
            tmpBuffer.data()[i] = 0;
        }
    }

    resultCrossNumber.save_image(resultPrefix + "resultCrossNumber.png");

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
    OclInfo ocl_info = OclInfo::init_opencl();
    DLOG("Opencl initialized");

    // init kernels
    ImgTransform img_transformer(ocl_info);
    ImgStatics img_statics(ocl_info);
    MinutiaeDetector detector(ocl_info);

    LOG("kernel loaded");

    // load image
    Img img1(pathPrefix + "101_3.tif");
    Img img2(pathPrefix + "101_4.tif");

    LOG("Image loaded");

    unique_ptr<MatrixBuffer<BYTE>> buffer1 = preprocess(
        img1, ocl_info, img_transformer, img_statics, detector, "img1_");
    unique_ptr<MatrixBuffer<BYTE>> buffer2 = preprocess(
        img2, ocl_info, img_transformer, img_statics, detector, "img2_");

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
    OclInfo ocl_info = OclInfo::init_opencl();
    DLOG("Opencl initialized");

    // init kernels
    ImgTransform img_transformer(ocl_info);
    ImgStatics img_statics(ocl_info);
    MinutiaeDetector detector(ocl_info);

    LOG("kernel loaded");

    // load image
    Img img1(pathPrefix + "101_3.tif");
    Img img2(pathPrefix + "101_4.tif");

    LOG("Image loaded");

    unique_ptr<MatrixBuffer<BYTE>> buffer1 = preprocess(
        img1, ocl_info, img_transformer, img_statics, detector, "img1_");
    unique_ptr<MatrixBuffer<BYTE>> buffer2 = preprocess(
        img2, ocl_info, img_transformer, img_statics, detector, "img2_");

    LOG("Image Preprocessed");

    FreeImage_DeInitialise();
}

}  // namespace driver
}  // namespace fingerprint_parallel

// driver code
int main(int argc, char** argv) {
    cout << argv[0] << endl;

    // fingerprint_parallel::driver::identicalRun();
    fingerprint_parallel::driver::run1();
    return 0;
}
