#pragma onece

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "CL/opencl.hpp"

extern "C" {
#include "FreeImage.h"
}

#include "OclInfo.hpp"
#include "MatrixBuffer.hpp"