
add_executable (
    FingerprintParallel 
    "main.cpp" 
    "main.hpp"
)

target_include_directories(FingerprintParallel PUBLIC "./")

target_link_libraries(
    FingerprintParallel PUBLIC
    FingerprintParallelCore 
)
