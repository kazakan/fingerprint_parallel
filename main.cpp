// CmakeOpenclManualLink.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include "main.hpp"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

string clErrorToStr(cl_int errorCode) {
    switch (errorCode) {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET"; //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; //-14
    case CL_COMPILE_PROGRAM_FAILURE:
        return "CL_COMPILE_PROGRAM_FAILURE"; //-15
    case CL_LINKER_NOT_AVAILABLE:
        return "CL_LINKER_NOT_AVAILABLE"; //-16
    case CL_LINK_PROGRAM_FAILURE:
        return "CL_LINK_PROGRAM_FAILURE"; //-17
    case CL_DEVICE_PARTITION_FAILED:
        return "CL_DEVICE_PARTITION_FAILED"; //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; //-19
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE"; //-63
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY"; //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return "CL_INVALID_IMAGE_DESCRIPTOR"; //-65
    case CL_INVALID_COMPILER_OPTIONS:
        return "CL_INVALID_COMPILER_OPTIONS"; //-66
    case CL_INVALID_LINKER_OPTIONS:
        return "CL_INVALID_LINKER_OPTIONS"; //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return "CL_INVALID_DEVICE_PARTITION_COUNT"; //-68
                                                    //    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
                                                    //    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70

    default:
        return "UNKNOWN ERROR CODE";
    }
}

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

class SingleChannelImgBuffer {
  private:
    shared_ptr<char[]> data = nullptr;
    unsigned int _width;
    unsigned int _height;
    unsigned int _len;

  public:
    SingleChannelImgBuffer(int width, int height)
        : _width(width), _height(height), _len(width * height) {
        data = shared_ptr<char[]>(new char[_len]);
    };
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