# Set source files
set(KERNEL_SOURCES
    distances.cu
    knn.cu
    lof.cu
    lof_cpp.cu
    utils.cu
)

# Create object library for CUDA kernels
add_library(cuda_lof_kernels OBJECT ${KERNEL_SOURCES})

# Set properties
set_target_properties(cuda_lof_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
) 