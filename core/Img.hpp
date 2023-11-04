#pragma once

#include <cstdint>
#include <exception>
#include <string>

extern "C" {
#include "FreeImage.h"
}
#include "MatrixBuffer.hpp"

class Img {
   public:
    std::size_t width;
    std::size_t height;
    std::size_t size;
    uint8_t *data;
    std::string path;

    enum ReadMode { GRAY, RGB };

    Img(std::string path) : path(path) {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);
        FIBITMAP *image = FreeImage_Load(format, path.c_str(), PNG_DEFAULT);
        if (image == nullptr) throw std::runtime_error("Cannot Load image");

        FIBITMAP *tmp = image;
        image = FreeImage_ConvertTo32Bits(image);
        FreeImage_Unload(tmp);

        this->width = FreeImage_GetWidth(image);
        this->height = FreeImage_GetHeight(image);
        this->size = width * height * 4;

        data = new uint8_t[size];
        memcpy(data, FreeImage_GetBits(image), size);

        FreeImage_Unload(image);
    }

    Img(MatrixBuffer<uint8_t> &matrixBuffer, ReadMode readMode = GRAY) {
        if (readMode == GRAY) {
            width = matrixBuffer.getWidth();
            height = matrixBuffer.getHeight();
            size = matrixBuffer.getLen() * 4;

            uint8_t *matDat = matrixBuffer.getData();

            data = new uint8_t[size];
            for (int i = 0; i < size / 4; ++i) {
                data[i * 4] = matDat[i];
                data[i * 4 + 1] = matDat[i];
                data[i * 4 + 2] = matDat[i];
                data[i * 4 + 3] = 255;
            }
        } else if (readMode == RGB) {
            width = matrixBuffer.getWidth() / 3;
            height = matrixBuffer.getHeight();
            size = width * height * 4;

            uint8_t *matDat = matrixBuffer.getData();

            data = new uint8_t[size];
            for (int i = 0; i < size / 4; ++i) {
                data[i * 4] = matDat[i * 3];
                data[i * 4 + 1] = matDat[i * 3 + 1];
                data[i * 4 + 2] = matDat[i * 3 + 2];
                data[i * 4 + 3] = 255;
            }
        }
    }

    ~Img() {
        delete[] data;
        data = nullptr;
    }

    bool saveImage(std::string filename) {
        return Img::saveImage(filename, data, width, height);
    }

    static bool saveImage(std::string fileName, uint8_t *buffer, int width,
                          int height) {
        FREE_IMAGE_FORMAT format =
            FreeImage_GetFIFFromFilename(fileName.c_str());
        FIBITMAP *image = FreeImage_ConvertFromRawBits(
            (uint8_t *)buffer, width, height, width * 4, 32, 0xFF000000,
            0x00FF0000, 0x0000FF00);
        return FreeImage_Save(format, image, fileName.c_str());
    }
};
