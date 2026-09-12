cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_lab VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CKLAB_ENABLE_CUDA "Build the CUDA backend if nvcc is available" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckLanguage)

set(CKLAB_HAS_CUDA OFF)
if(CKLAB_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        set(CKLAB_HAS_CUDA ON)
        message(STATUS "CUDA Kernel Lab: CUDA enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "CUDA Kernel Lab: nvcc not found; building host-only")
    endif()
else()
    message(STATUS "CUDA Kernel Lab: CUDA disabled (CKLAB_ENABLE_CUDA=OFF)")
endif()

set(CKLAB_COMMON
    src/cpu.cpp
    src/verify.cpp
    src/bench.cpp
)

add_library(cklab_core STATIC ${CKLAB_COMMON})
target_include_directories(cklab_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(cklab_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(cklab_core PRIVATE /W4 /WX)
else()
    target_compile_options(cklab_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(CKLAB_HAS_CUDA)
    add_library(cklab_cuda STATIC src/cuda_kernels.cu)
    target_include_directories(cklab_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(cklab_cuda PUBLIC cxx_std_17)
    target_compile_definitions(cklab_cuda PUBLIC CKLAB_HAS_CUDA=1)
    set_target_properties(cklab_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(cklab_core PUBLIC cklab_cuda)
    target_compile_definitions(cklab_core PUBLIC CKLAB_HAS_CUDA=1)
else()
    target_sources(cklab_core PRIVATE src/cuda_stub.cpp)
endif()

add_executable(cklab src/main.cpp)
target_link_libraries(cklab PRIVATE cklab_core)

add_executable(test_cklab tests/test_host.cpp)
target_link_libraries(test_cklab PRIVATE cklab_core)

enable_testing()
add_test(NAME host_unit COMMAND test_cklab)
add_test(NAME host_bench COMMAND cklab --mode host --n 48 --reps 2 --op all)
add_test(NAME cli_help COMMAND cklab --help)

set_tests_properties(host_bench PROPERTIES PASS_REGULAR_EXPRESSION "All checks passed")
