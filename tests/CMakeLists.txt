# Find Google Test
find_package(GTest)
if(NOT GTEST_FOUND)
    # If GTest is not found, download and build it
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
endif()

# Define test sources
set(TEST_SOURCES
    test_lof.cu
    test_distances.cu
    test_knn.cu
)

# Define test targets
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
        cuda_lof
        GTest::GTest
        GTest::Main
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Copy test data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Add a custom test target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_TARGETS}
) 