set(MODULE_NAME "core")

file(GLOB SRC "*.cpp" "*.hpp")

set(OCL_MODULE_SRC_NAME "ocl_${MODULE_NAME}_src")
file(GLOB OCL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/opencl/*.cl")

add_custom_command(
    OUTPUT "${OCL_MODULE_SRC_NAME}.cpp" "${OCL_MODULE_SRC_NAME}.hpp"
    COMMAND ${CMAKE_COMMAND}
    "-DMODULE_NAME=${MODULE_NAME}" 
    "-DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DOCL_KERNELS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/opencl"
    
    -P "${PROJECT_SOURCE_DIR}/cmake/OclToCpp.cmake"
    DEPEND "${PROJECT_SOURCE_DIR}/cmake/OclToCpp.cmake" 
    "${CMAKE_CURRENT_SOURCE_DIR}/opencl"
)

add_library(
    FingerprintParallelCore STATIC 
    ${SRC} 
    "${OCL_MODULE_SRC_NAME}.cpp" 
    "${OCL_MODULE_SRC_NAME}.hpp"
)

target_include_directories(FingerprintParallelCore PUBLIC "./")
target_include_directories(FingerprintParallelCore PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

find_package(OpenCL REQUIRED)
target_link_libraries(FingerprintParallelCore OpenCL::OpenCL)

find_library(FreeImage PUBLIC_HEADER)
target_link_libraries(FingerprintParallelCore FreeImage)

