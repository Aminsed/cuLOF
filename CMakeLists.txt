cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Find nvcc before the CUDA language is enabled.
#
# CMake searches only PATH for nvcc. The usual Linux install puts the toolkit
# under /usr/local/cuda and only the driver on PATH, which would otherwise be
# indistinguishable from having no CUDA at all -- this is why `pip install`
# failed on machines that had a perfectly good toolkit.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    file(GLOB _cuda_roots /usr/local/cuda-*)
    find_program(_nvcc nvcc
        HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT
              /usr/local/cuda ${_cuda_roots}
        PATH_SUFFIXES bin)
    if(_nvcc)
        set(CMAKE_CUDA_COMPILER "${_nvcc}")
    endif()
endif()

project(culof LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(CULOF_BUILD_PYTHON "Build the Python extension module" ON)
option(CULOF_BUILD_TESTS  "Build the C++/CUDA unit tests"     OFF)

# ---------------------------------------------------------------------------
# CUDA architectures
#
# Order: -DCULOF_CUDA_ARCHITECTURES, then $CUDAARCHS, then the local GPU, then a
# portable fat binary.
#
# Note the absence of `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`: project()
# already seeds that variable, so such a guard never fires. That is precisely
# how this project used to compile for sm_52 on every machine, including an
# sm_86 A6000.
# ---------------------------------------------------------------------------
set(CULOF_CUDA_ARCHITECTURES "" CACHE STRING
    "Target architectures, e.g. '86' or '80;86;89'. Empty means autodetect.")

if(CULOF_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CULOF_CUDA_ARCHITECTURES})
elseif(DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
else()
    set(_detected "")
    find_program(CULOF_NVIDIA_SMI nvidia-smi)
    if(CULOF_NVIDIA_SMI)
        execute_process(
            COMMAND ${CULOF_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _caps RESULT_VARIABLE _rc
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_rc EQUAL 0 AND _caps)
            string(REGEX MATCHALL "[0-9]+\\.[0-9]+" _list "${_caps}")
            foreach(_cap IN LISTS _list)
                string(REPLACE "." "" _cap "${_cap}")
                list(APPEND _detected ${_cap})
            endforeach()
            list(REMOVE_DUPLICATES _detected)
        endif()
    endif()
    # No GPU visible (CI, container, cross-build): cover Volta through Hopper.
    set(CMAKE_CUDA_ARCHITECTURES "${_detected}")
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

# Deliberately no --use_fast_math. This library's contract is agreement with
# scikit-learn to within float32 rounding, and fast-math's reciprocal and
# square-root approximations break that to buy a few percent.
add_library(culof_core STATIC
    src/cuda/preprocess.cu
    src/cuda/knn.cu
    src/cuda/lof.cu
    src/cuda/pipeline.cu
)
target_include_directories(culof_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(culof_core PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(culof_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

# ---------------------------------------------------------------------------
# Python module
#
# pybind11 >= 2.12 is a hard floor. Earlier releases read NumPy's internal
# descriptor layout, which changed in NumPy 2.0; the combination returns arrays
# with a stride of 0, so every element aliases element 0 and the caller silently
# receives a constant. This project shipped exactly that bug. The floor is what
# stops it returning.
# ---------------------------------------------------------------------------
if(CULOF_BUILD_PYTHON)
    set(CULOF_MIN_PYBIND11 2.12)
    find_package(pybind11 ${CULOF_MIN_PYBIND11} CONFIG QUIET)

    if(NOT pybind11_FOUND)
        find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                    "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _pybind_dir RESULT_VARIABLE _pybind_rc
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_pybind_rc EQUAL 0 AND _pybind_dir)
            find_package(pybind11 ${CULOF_MIN_PYBIND11} CONFIG QUIET
                         PATHS "${_pybind_dir}" NO_DEFAULT_PATH)
        endif()
    endif()

    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6 GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_culof src/cuda/binding.cpp)
    target_link_libraries(_culof PRIVATE culof_core)
    install(TARGETS _culof LIBRARY DESTINATION culof)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CULOF_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0 GIT_SHALLOW TRUE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(culof_tests tests/cpp/test_culof.cu)
    target_link_libraries(culof_tests PRIVATE culof_core GTest::gtest GTest::gtest_main)
    add_test(NAME culof_tests COMMAND culof_tests)
endif()

message(STATUS "cuLOF: CUDA ${CUDAToolkit_VERSION}, "
               "arch ${CMAKE_CUDA_ARCHITECTURES}, ${CMAKE_BUILD_TYPE}")
