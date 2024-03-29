if(NOT DEFINED MODULE_NAME)
    message(FATAL_ERROR "MODULE_NAME is not defined")
else()
    message("Processing for module ${MODULE_NAME}")
endif()



if(NOT DEFINED OCL_KERNELS_DIR)
    message(FATAL_ERROR "OCL_KERNELS_DIR is not defined")
else()
    message("OCL_KERNELS_DIR=${OCL_KERNELS_DIR}")
endif()

if(NOT DEFINED OUT_DIR)
    message(FATAL_ERROR "OUT_DIR is not defined")
else()
    message("OUT_DIR=${OUT_DIR}")
endif()

file(GLOB OCL_FILES "${OCL_KERNELS_DIR}/*.cl")

if(NOT OCL_FILES)
    message(FATAL_ERROR "No .cl file in folder ${CURRENT_MODULE_PATH}/opencl")
endif()

set(OCL_MODULE_SRC_NAME "ocl_${MODULE_NAME}_src")
 
set(HEADER_GUARD_VAR "${MODULE_NAME}_OPENCL_AUTOGENERATED_OCL_${MODULE_NAME}_SRC_HPP")
string(TOUPPER "${HEADER_GUARD_VAR}" HEADER_GUARD_VAR)

set(SRC_HPP "#ifndef ${HEADER_GUARD_VAR}\n")
string(APPEND SRC_HPP "#define ${HEADER_GUARD_VAR}\n")

set(SRC_CPP "#include \"${OCL_MODULE_SRC_NAME}.hpp\"\n")

foreach(cl ${OCL_FILES})
    file(READ "${cl}" OCL_CODE_LINES)

    string(REGEX REPLACE "/\\*([^/*]|[^*]/|\\*[^/])*\\*/" "" OCL_CODE_LINES "${OCL_CODE_LINES}") 
    string(REGEX REPLACE "//[^\n]*\n" "\n" OCL_CODE_LINES "${OCL_CODE_LINES}")

    string(REGEX REPLACE "\n([ ]*\n)+" "\n" OCL_CODE_LINES "${OCL_CODE_LINES}")

    string(REPLACE "\\" "\\\\" OCL_CODE_LINES "${OCL_CODE_LINES}")
    string(REPLACE "\"" "\\\*" OCL_CODE_LINES "${OCL_CODE_LINES}")
    string(REPLACE "\n" "\\n\"\n\"" OCL_CODE_LINES "${OCL_CODE_LINES}")

    
    get_filename_component(OCL_FILE_NAME ${cl} NAME_WE)
        
    set(SRC_DECLARE "extern const char* ocl_src_${OCL_FILE_NAME};\n")
    set(SRC_DEFINITION "const char* ocl_src_${OCL_FILE_NAME} = \"${OCL_CODE_LINES}\";\n")

    set(SRC_HPP "${SRC_HPP}${SRC_DECLARE}")
    set(SRC_CPP "${SRC_CPP}${SRC_DEFINITION}")
endforeach()

string(APPEND SRC_HPP "#endif // ${HEADER_GUARD_VAR}\n")

file(WRITE "${OUT_DIR}/${OCL_MODULE_SRC_NAME}.hpp" "${SRC_HPP}")
file(WRITE "${OUT_DIR}/${OCL_MODULE_SRC_NAME}.cpp" "${SRC_CPP}")

