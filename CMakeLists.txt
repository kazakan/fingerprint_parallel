
cmake_minimum_required (VERSION 3.1)

project ("FingerprintParallel" LANGUAGES CXX)

add_executable (run "main.cpp" "main.hpp")

# Library Path Setting
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})

# OPENCL

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 

LINK_DIRECTORIES(${OpenCL_LIBRARY})

target_link_libraries(run OpenCL::OpenCL)

# FreeImage
find_library(FREEIMAGE_LIB NAMES FreeImage PATHS ${LIBRARY_DIRS} REQUIRED)
message("Found FreeImage Libriary : ${FREEIMAGE_LIB}")
find_path(FREEIMAGE_INCLUDE_DIRS NAMES FreeImage.h PATHS $ENV{CPATH} REQUIRED)

target_include_directories(run PRIVATE ${FREEIMAGE_INCLUDE_DIRS})
target_link_libraries(run ${FREEIMAGE_LIB})

# prevent std segfault
target_link_options(run PRIVATE -static-libgcc -static-libstdc++ -static)

