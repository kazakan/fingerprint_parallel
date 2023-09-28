#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <vector>

#include "CL/opencl.hpp"
#include "OclException.hpp"
#include "logger.hpp"

class OclInfo {
   public:
    std::vector<cl::Platform> platforms;
    cl::Context ctx;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;

    static OclInfo initOpenCL(bool useGpu = true) {
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        if (platformList.size() == 0) {
            DLOG("No available Opencl Platform.")
        }
        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};

        cl::Context ctx(useGpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU,
                        cprops);

        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

        if (devices.size() == 0) {
            DLOG("No available Opencl Devices.")
        }
        cl::CommandQueue queue(ctx, devices[0], 0);

        OclInfo oclinfo = {platformList, ctx, devices, queue};

        return oclinfo;
    }

    static void showPlatformInfos() {
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        for (cl::Platform platform : platformList) {
            std::string profile_result;
            std::string version_result;
            std::string name_result;
            std::string vender_result;
            std::string extensions_result;
            cl_ulong host_timer_resolution_result = 0L;
            std::string icd_suffix_khr_result;

            platform.getInfo(CL_PLATFORM_PROFILE, &profile_result);
            platform.getInfo(CL_PLATFORM_VERSION, &version_result);
            platform.getInfo(CL_PLATFORM_NAME, &name_result);
            platform.getInfo(CL_PLATFORM_VENDOR, &vender_result);
            //platform.getInfo(CL_PLATFORM_EXTENSIONS, &extensions_result);
            platform.getInfo(CL_PLATFORM_HOST_TIMER_RESOLUTION,
                             &host_timer_resolution_result);
            platform.getInfo(CL_PLATFORM_ICD_SUFFIX_KHR,
                             &icd_suffix_khr_result);

            LOG("========== Platform %s info ==========", name_result.data());
            LOG("Name : %s", name_result.data());
            LOG("PLATFORM_PROFILE : %s", profile_result.data());
            LOG("Version : %s", version_result.data());
            LOG("Vendor : %s", vender_result.data());
            //LOG("Extensions : %s", extensions_result.data());
            LOG("Host timer resolution : %ul", host_timer_resolution_result);
            LOG("ICD suffix KHR : %s", icd_suffix_khr_result.data());

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (const cl::Device& device : devices) {
                std::string dev_name_result;
                cl_device_type dtype;
                size_t max_workgroup_size=0;
                device.getInfo(CL_DEVICE_NAME, &dev_name_result);
                device.getInfo(CL_DEVICE_TYPE, &dtype);
                device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_workgroup_size);

                LOG("========== Device %s info ==========",
                    dev_name_result.data());
                LOG("Name : %s", dev_name_result.data());
                LOG("Type : %s",
                    dtype == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU or else");
                    
                LOG("Max workgroup size : %lld", max_workgroup_size);
                LOG("========== End of device %s ==========",
                    dev_name_result.data());
            }
            LOG("========== End of platform %s ==========", name_result.data());
        }
    }
};