#include <memory>
#include "CL/opencl.hpp"

class MatrixBuffer {
  private:
    std::shared_ptr<char[]> data = nullptr;
    unsigned int _width;
    unsigned int _height;
    unsigned int _len;

  public:
    MatrixBuffer(int width, int height)
        : _width(width), _height(height), _len(width * height) {
        data = std::shared_ptr<char[]>(new char[_len]);
    };

    
};
