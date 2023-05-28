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
    char *data;
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

        data = new char[size];
        memcpy(data, FreeImage_GetBits(image), size);

        FreeImage_Unload(image);
    }

    ~Img() {
        delete[] data;
        data = nullptr;
    }

    static bool saveImage(string fileName, char *buffer, int width, int height) {
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
    cl::ImageFormat outImgFormat(CL_RGBA, CL_UNSIGNED_INT8);

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

    char *outImgData = new char[img.size];

    cl::Image2D outImg(
        oclinfo.ctx,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        outImgFormat,
        img.width, img.height, 0, outImgData, &err);
    checkErr(err, 2);

    // create kernel from source code
    cl::Program::Sources sources;
    sources.push_back(sourceStr);
    cl::Program program(oclinfo.ctx, sources);
    err = program.build(oclinfo.devices);
    checkErr(err, 3);

    cl::Kernel kernel(program, "gray");
    cl::Kernel dynamicThresholdKernel(program, "dynamicThresholding");

    // set kernel arg

    kernel.setArg(0, climg);
    kernel.setArg(1, outImg);

    // dynamic thresholding
    dynamicThresholdKernel.setArg(0, outImg);
    dynamicThresholdKernel.setArg(1, climg);
    dynamicThresholdKernel.setArg(2, 9);

    const size_t wsize = 16;
    cl::NDRange local_work_size(wsize, wsize);
    cl::NDRange global_work_size(img.width, img.height);

    // launch kernel
    err = oclinfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
    checkErr(err, 4);
    err = oclinfo.queue.enqueueNDRangeKernel(dynamicThresholdKernel, cl::NullRange, global_work_size, local_work_size);
    checkErr(err, 5);

    // ger return value
    char *outBuffer = (char *)oclinfo.queue.enqueueMapImage(climg, CL_TRUE, CL_MAP_READ, {0, 0, 0}, {img.width, img.height, 1}, 0, NULL, NULL, NULL, &err);
    checkErr(err, 6);

    // write image

    Img::saveImage(pathPrefix + "result.png", outBuffer, img.width, img.height);
    FreeImage_DeInitialise();

    // delete [] outImgData;

    return 0;
}