#ifndef FINGERPRINT_PARALLEL_CORE_IMG_HPP_
#define FINGERPRINT_PARALLEL_CORE_IMG_HPP_

#include <cstdint>
#include <exception>
#include <string>

extern "C" {
#include "FreeImage.h"
}
#include "MatrixBuffer.hpp"

namespace fingerprint_parallel {
namespace core {

class Img {
   public:
    enum ReadMode { GRAY, RGB };

    Img(std::string path) : path_(path) {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);
        FIBITMAP *image = FreeImage_Load(format, path.c_str(), PNG_DEFAULT);
        if (image == nullptr) throw std::runtime_error("Cannot Load image");

        FIBITMAP *tmp = image;
        image = FreeImage_ConvertTo32Bits(image);
        FreeImage_Unload(tmp);

        width_ = FreeImage_GetWidth(image);
        height_ = FreeImage_GetHeight(image);
        size_ = width_ * height_ * 4;

        data_ = new uint8_t[size_];
        memcpy(data_, FreeImage_GetBits(image), size_);

        FreeImage_Unload(image);
    }

    Img(MatrixBuffer<uint8_t> &matrix_buffer, ReadMode read_mode = GRAY) {
        if (read_mode == GRAY) {
            width_ = matrix_buffer.width();
            height_ = matrix_buffer.height();
            size_ = matrix_buffer.size() * 4;

            uint8_t *mat_dat = matrix_buffer.data();

            data_ = new uint8_t[size_];
            for (int i = 0; i < size_ / 4; ++i) {
                data_[i * 4] = mat_dat[i];
                data_[i * 4 + 1] = mat_dat[i];
                data_[i * 4 + 2] = mat_dat[i];
                data_[i * 4 + 3] = 255;
            }
        } else if (read_mode == RGB) {
            width_ = matrix_buffer.width() / 3;
            height_ = matrix_buffer.height();
            size_ = width_ * height_ * 4;

            uint8_t *mat_dat = matrix_buffer.data();

            data_ = new uint8_t[size_];
            for (int i = 0; i < size_ / 4; ++i) {
                data_[i * 4] = mat_dat[i * 3];
                data_[i * 4 + 1] = mat_dat[i * 3 + 1];
                data_[i * 4 + 2] = mat_dat[i * 3 + 2];
                data_[i * 4 + 3] = 255;
            }
        }
    }

    ~Img() {
        delete[] data_;
        data_ = nullptr;
    }

    bool save_image(std::string file_name) {
        return Img::save_image(file_name, data_, width_, height_);
    }

    static bool save_image(std::string file_name, uint8_t *buffer, int width,
                           int height) {
        FREE_IMAGE_FORMAT format =
            FreeImage_GetFIFFromFilename(file_name.c_str());
        FIBITMAP *image = FreeImage_ConvertFromRawBits(
            (uint8_t *)buffer, width, height, width * 4, 32, 0xFF000000,
            0x00FF0000, 0x0000FF00);
        return FreeImage_Save(format, image, file_name.c_str());
    }

    const std::size_t width() const { return width_; }

    const std::size_t height() const { return height_; }

    const std::size_t size() const { return size_; }

    uint8_t *data() { return data_; }

   private:
    std::size_t width_;
    std::size_t height_;
    std::size_t size_;
    uint8_t *data_;
    std::string path_;
};

}  // namespace core
}  // namespace fingerprint_parallel

#endif  // FINGERPRINT_PARALLEL_CORE_IMG_HPP_