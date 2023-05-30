#pragma onece

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "CL/opencl.hpp"

extern "C" {
#include "FreeImage.h"
}

#include "Img.hpp"
#include "MatrixBuffer.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"
#include "ImgTransform.hpp"
#include "ImgStatics.hpp"