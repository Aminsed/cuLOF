# Find Python
find_package(Python COMPONENTS Interpreter Development)

# Find pybind11
if(NOT TARGET pybind11::pybind11)
    # If pybind11 is not found, download and build it
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add the Python module
pybind11_add_module(cuda_lof_py
    lof_binding.cpp
)

# Set CUDA_RESOLVE_DEVICE_SYMBOLS to resolve CUDA symbols at link time
set_target_properties(cuda_lof_py PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "_cuda_lof"
)

# Find CUDA libraries explicitly
find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
find_library(CUDA_CURAND_LIBRARY curand PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)

# Link to the CUDA LOF library and CUDA runtime
target_link_libraries(cuda_lof_py PRIVATE
    cuda_lof
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
)

# Make sure the CUDA runtime is linked properly
set_target_properties(cuda_lof_py PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed -lcudart -lcuda"
)

# Install the Python module
install(TARGETS cuda_lof_py
    LIBRARY DESTINATION "${Python_SITEARCH}"
)

# Install the Python package files
install(FILES
    __init__.py
    DESTINATION "${Python_SITEARCH}/cuda_lof"
) 