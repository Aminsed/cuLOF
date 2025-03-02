# Function to detect CUDA architectures
function(detect_cuda_architectures output_var)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # Check if CMAKE_CUDA_ARCHITECTURES is already defined
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            message(STATUS "Using predefined CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
            set(${output_var} ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
            return()
        endif()

        # Try to detect the architecture of the currently installed GPU
        set(DETECT_ARCH_FILE ${CMAKE_BINARY_DIR}/detect_cuda_arch.cu)
        file(WRITE ${DETECT_ARCH_FILE} "
            #include <iostream>
            int main() {
                int device_count = 0;
                if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
                    std::cerr << \"Failed to get CUDA device count\" << std::endl;
                    return 1;
                }
                if (device_count == 0) {
                    std::cerr << \"No CUDA devices found\" << std::endl;
                    return 1;
                }
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
                    std::cerr << \"Failed to get CUDA device properties\" << std::endl;
                    return 1;
                }
                int major = prop.major;
                int minor = prop.minor;
                std::cout << major << minor << std::endl;
                return 0;
            }
        ")

        set(DETECT_ARCH_EXE ${CMAKE_BINARY_DIR}/detect_cuda_arch)
        try_run(
            RUN_RESULT
            COMPILE_RESULT
            ${CMAKE_BINARY_DIR}
            ${DETECT_ARCH_FILE}
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
            LINK_LIBRARIES ${CUDA_LIBRARIES}
            RUN_OUTPUT_VARIABLE ARCH
        )

        # Remove the temporary files
        file(REMOVE ${DETECT_ARCH_FILE} ${DETECT_ARCH_EXE})

        if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
            # Successfully detected the architecture
            string(STRIP "${ARCH}" ARCH)
            set(DETECTED_ARCH ${ARCH})
            message(STATUS "Detected CUDA architecture: ${DETECTED_ARCH}")
            set(${output_var} ${DETECTED_ARCH} PARENT_SCOPE)
        else()
            # Fallback to a reasonable default
            set(FALLBACK_ARCH "75")
            message(STATUS "Failed to detect CUDA architecture, using fallback: ${FALLBACK_ARCH}")
            set(${output_var} ${FALLBACK_ARCH} PARENT_SCOPE)
        endif()
    else()
        message(STATUS "Not using NVIDIA CUDA compiler, using default architecture")
        set(${output_var} "75" PARENT_SCOPE)
    endif()
endfunction() 