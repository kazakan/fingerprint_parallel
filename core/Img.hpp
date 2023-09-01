#pragma once

#include <exception>
#include <string>

extern "C" {
#include "FreeImage.h"
}
#include "MatrixBuffer.hpp"

using namespace std;

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
        if (image == nullptr) throw runtime_error("Cannot Load image");

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

    Img(MatrixBuffer<BYTE> &matrixBuffer) {
        width = matrixBuffer.getWidth();
        height = matrixBuffer.getHeight();
        size = matrixBuffer.getLen() * 4;

        unsigned char *matDat = matrixBuffer.getData();

        data = new unsigned char[size];
        for (int i = 0; i < size / 4; ++i) {
            data[i * 4] = matDat[i];
            data[i * 4 + 1] = matDat[i];
            data[i * 4 + 2] = matDat[i];
            data[i * 4 + 3] = 255;
        }
    }

    ~Img() {
        delete[] data;
        data = nullptr;
    }

    bool saveImage(string filename) {
        return Img::saveImage(filename, data, width, height);
    }

    static bool saveImage(string fileName, unsigned char *buffer, int width,
                          int height) {
        FREE_IMAGE_FORMAT format =
            FreeImage_GetFIFFromFilename(fileName.c_str());
        FIBITMAP *image = FreeImage_ConvertFromRawBits(
            (BYTE *)buffer, width, height, width * 4, 32, 0xFF000000,
            0x00FF0000, 0x0000FF00);
        return FreeImage_Save(format, image, fileName.c_str());
    }
};
