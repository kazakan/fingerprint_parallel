#include <time.h>

#include <CL/opencl.hpp>

#include "ImgStatics.hpp"
#include "OclInfo.hpp"
#include "ScalarBuffer.hpp"
#include "ocl_core_src.hpp"
#include "random_case_generator.hpp"


using namespace fingerprint_parallel::core;

int main(void) {
    OclInfo::showPlatformInfos();
    OclInfo ocl_info = OclInfo::init_opencl();

    DLOG("Opencl initialized");

    ImgStatics img_statics(ocl_info);

    double total_time = 0;
    RandomMatrixGenerator generator;

    cl::Program::Sources sources;
    sources.push_back(ocl_src_statics);
cl:;
    cl::Program program = cl::Program(ocl_info.ctx_, sources);

    cl_int err = program.build(ocl_info.devices_);
    if (err) throw OclBuildException(err);

    cl::Kernel kernel_sum(program, "sum_uchar_long");

    for (int i = 0; i < 1000; ++i) {
        std::tuple<int, int, std::vector<uint8_t>> data =
            generator.generate_matrix_data(0, 255);

        MatrixBuffer<uint8_t> buffer_original(
            std::get<0>(data), std::get<1>(data), std::get<2>(data));
        buffer_original.create_buffer(&ocl_info);
        buffer_original.to_gpu();

        ScalarBuffer<uint64_t> buffer_result;
        buffer_result.create_buffer(&ocl_info);

        const int N = buffer_original.size();
        const int group_size = 512;
        const int n_groups = (N + (group_size - 1)) / group_size;
        const int n_threads = n_groups * group_size;

        MatrixBuffer<uint64_t> tmp1(group_size, 1);
        tmp1.create_buffer(buffer_original.ocl_info());

        kernel_sum.setArg(0, *buffer_original.buffer());
        kernel_sum.setArg(1, *buffer_result.buffer());
        kernel_sum.setArg(2, group_size * sizeof(int64_t), NULL);
        kernel_sum.setArg(3, N);

        cl::Event e;
        cl_int err = ocl_info.queue_.enqueueNDRangeKernel(
            kernel_sum, cl::NullRange, cl::NDRange(group_size),
            cl::NDRange(group_size), nullptr, &e);
        if (err) throw OclKernelEnqueueError(err);

        e.wait();
        
        cl_ulong start = 0, end = 0;
        e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);

        total_time += (cl_double)(end - start) * (cl_double)(1e-06);
        
    }

    DLOG("%llf ms total.", total_time);
}