// CmakeOpenclManualLink.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include "main.hpp"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

class Img {
  public:
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned char *data;
    string path;

    Img(string path) : path(path) {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);
        FIBITMAP *image = FreeImage_Load(format, path.c_str(), PNG_DEFAULT);
        if (image == nullptr)
            throw runtime_error("Cannot Load image");

        FIBITMAP *tmp = image;
        image = FreeImage_ConvertTo32Bits(image);
        FreeImage_Unload(tmp);

        this->width = FreeImage_GetWidth(image);
        this->height = FreeImage_GetHeight(image);
        this->size = width * height * 4;

        data = new unsigned char[size];
        memcpy(data, FreeImage_GetBits(image), size);

        FreeImage_Unload(image);
    }

    Img(MatrixBuffer& matrixBuffer){
        width = matrixBuffer.getWidth();
        height = matrixBuffer.getHeight();
        size = matrixBuffer.getLen()*4;

        unsigned char* matDat = matrixBuffer.getData();

        data = new unsigned char[size];
        for(int i=0;i< size/4 ;++i){
            data[i*4] = matDat[i];
            data[i*4+1] = matDat[i];
            data[i*4+2] = matDat[i];
            data[i*4+3] = 255;
        }
    }

    ~Img() {
        delete[] data;
        data = nullptr;
    }

    bool saveImage(string filename){
        return Img::saveImage(filename,data,width,height);
    }

    static bool saveImage(string fileName, unsigned char *buffer, int width, int height) {
        FREE_IMAGE_FORMAT format =
            FreeImage_GetFIFFromFilename(fileName.c_str());
        FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE *)buffer,
                                                       width,
                                                       height, width * 4, 32,
                                                       0xFF000000, 0x00FF0000, 0x0000FF00);
        return FreeImage_Save(format, image, fileName.c_str());
    }
};


string readFile(string path) {
    ifstream ifs(path);
    return string((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));
}


int checkErr(int err, int id = 0) {
    if (err != CL_SUCCESS)
        cout << "Error : " + clErrorToStr(err) << " pos :" << id << endl;
    return err;
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
    MatrixBuffer buffer1(img.width,img.height);
    MatrixBuffer buffer2(img.width,img.height);

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
    checkErr(err, 1);

    // create kernel from source code
    cl::Program::Sources sources;
    sources.push_back(sourceStr);
    cl::Program program(oclinfo.ctx, sources);
    err = program.build(oclinfo.devices);
    checkErr(err, 3);

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
    checkErr(err, 4);

    // ger return value
    err = oclinfo.queue.enqueueReadBuffer(*buffer1.getClBuffer(), CL_TRUE,0,buffer1.getLen(),buffer1.getData(),nullptr,nullptr);
    checkErr(err, 5);
    Img resultImg1(buffer1);
    bool saved = resultImg1.saveImage(pathPrefix+"result1.png");
    if(!saved){
        cerr<<"Failed save image"<<endl;
    }

    // run kernel 
    err = oclinfo.queue.enqueueNDRangeKernel(dynamicThresholdKernel, cl::NullRange, global_work_size, local_work_size);
    checkErr(err, 6);

    // ger return value
    err = oclinfo.queue.enqueueReadBuffer(*buffer2.getClBuffer(), CL_TRUE,0,buffer2.getLen(),buffer2.getData(),nullptr,nullptr);
    checkErr(err, 7);

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