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
#include "ImgStatics.hpp"
#include "ImgTransform.hpp"
#include "MatrixBuffer.hpp"
#include "MinutiaeDetector.hpp"
#include "OclException.hpp"
#include "OclInfo.hpp"
#include "logger.hpp"