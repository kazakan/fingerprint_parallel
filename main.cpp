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
    ios::sync_with_stdio(0);
    cout.tie(0);

    string pathPrefix = "../";
    cl_int err = 0;
    FreeImage_Initialise(true);

    cout << argv[0] << endl;

    cout << "Running" << endl;
    // load image

    Img img(pathPrefix + "fingerprint.BMP");
    cout << "Loaded Image" << endl;

    // init opencl
    OclInfo oclInfo = OclInfo::initOpenCL();
    cout<<"Opencl initialized"<<endl;

    // Load kernel
    string transformSource = readFile(pathPrefix + "transform.cl");
    string staticsSoure = readFile(pathPrefix + "statics.cl");
    cout << "kernel file loadded" << endl;

    ImgTransform imgTransformer(oclInfo,transformSource);
    ImgStatics imgStatics(oclInfo,staticsSoure);
    MinutiaeDetector detector(oclInfo,transformSource);

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
    buffer1.toHost(oclInfo);
    Img resultGray(buffer1);
    resultGray.saveImage(pathPrefix + "resultGray.png");

    // normalize
    float mean = imgStatics.mean(buffer1);
    float var = imgStatics.var(buffer1);

    cout<<"Mean : "<<mean<<" var : "<<var<<endl;

    imgTransformer.normalize(buffer1,buffer2,128,2000,mean,var);
    buffer2.toHost(oclInfo);
    Img resultNormalize(buffer2);
    resultNormalize.saveImage(pathPrefix + "resultNormalize.png");

    // gaussian filter
    imgTransformer.applyGaussianFilter(buffer2,buffer1);

    buffer1.toHost(oclInfo);
    Img resultGaussian(buffer1);
    resultGaussian.saveImage(pathPrefix + "resultGaussian.png");

    // dynamic thresholding
    imgTransformer.applyDynamicThresholding(buffer1,buffer2,3);

    buffer2.toHost(oclInfo);
    Img resultThreshold(buffer2);
    resultThreshold.saveImage(pathPrefix + "resultThreshold.png");

    // thinning
    imgTransformer.applyThinning8(buffer2,buffer1);

    buffer1.toHost(oclInfo);
    Img resultThinning(buffer1);
    resultThinning.saveImage(pathPrefix + "resultThinning.png");

    // cross number
    detector.applyCrossNumber(buffer1,buffer2);
    buffer2.toHost(oclInfo);
    Img resultCrossNum(buffer2);
    resultCrossNum.saveImage(pathPrefix + "resultCrossNum.png");

    
    cout<<buffer2.getLen()<<endl;
    for(int i=0;i<buffer2.getLen();++i){
        BYTE val = buffer2.getData()[i];
        if(val != 0){
            cout<<"Found type "<< (int )val << " at "<<i<<"\n";
        }
    }

    
    FreeImage_DeInitialise();

    // delete [] outImgData;

    return 0;
}