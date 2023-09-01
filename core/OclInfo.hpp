#pragma once

#include <vector>

#include "CL/opencl.hpp"

class OclInfo {
   public:
    std::vector<cl::Platform> platforms;
    cl::Context ctx;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;

    static OclInfo initOpenCL() {
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};

        cl::Context ctx(CL_DEVICE_TYPE_GPU, cprops);

        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(ctx, devices[0], 0);

        OclInfo oclinfo = {platformList, ctx, devices, queue};

        return oclinfo;
    }
};