
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  unit_test
  kernel_test.cpp
  img_statics_test.cpp
  minutiae_extract_test.cpp
  random_case_generator.hpp
)

target_link_libraries(
  unit_test
  GTest::gtest_main 
  FingerprintParallelCore
)

include(GoogleTest)
gtest_discover_tests(unit_test)


add_executable(
    reduction_time_test
    reduction_alg_time_test.cpp
    random_case_generator.hpp
)

target_link_libraries(
    reduction_time_test   
    FingerprintParallelCore
)