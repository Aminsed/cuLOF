# Add CUDA source directory
add_subdirectory(cuda)

# Add the CUDA LOF library
add_library(cuda_lof STATIC
    cuda/distances.cu
    cuda/knn.cu
    cuda/lof.cu
    cuda/utils.cu
    cuda/lof_cpp.cu
)

# Set CUDA properties
set_target_properties(cuda_lof PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Find CUDA libraries explicitly
find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
find_library(CUDA_CURAND_LIBRARY curand PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)

# Link against CUDA libraries
target_link_libraries(cuda_lof
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
)

# Include directories
target_include_directories(cuda_lof PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Export symbols
target_compile_definitions(cuda_lof PRIVATE CUDA_LOF_EXPORTS)

# Install the library
install(TARGETS cuda_lof
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
) 