cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(UnifiedFPN LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# Optional packages
find_package(PythonInterp 3.8 REQUIRED)
find_package(pybind11 CONFIG)
find_package(GTest)

# CUDA setup
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures - support modern GPUs
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
endif()

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -Xptxas=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# Source files
set(CUDA_SOURCES
    src/kernels/unified_fpn_kernels.cu
    src/core/fpn_core.cu
)

set(CXX_SOURCES
    src/fpn_interface.cpp
)

set(HEADER_FILES
    include/fpn_types.h
    include/fpn_kernels.h
)

# Main library
add_library(unified_fpn SHARED
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
    ${HEADER_FILES}
)

# Link libraries
target_link_libraries(unified_fpn
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
    CUDA::curand
)

# Set properties
set_target_properties(unified_fpn PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "75;80;86;89"
)

# Installation
install(TARGETS unified_fpn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADER_FILES}
    DESTINATION include/unified_fpn
)

# Benchmark executable
add_executable(fpn_benchmark
    benchmarks/fpn_benchmark.cu
    ${CUDA_SOURCES}
)

target_link_libraries(fpn_benchmark
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
    CUDA::curand
)

set_target_properties(fpn_benchmark PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "75;80;86;89"
)

# Testing
if(GTest_FOUND)
    enable_testing()
    
    add_executable(fpn_tests
        tests/test_fpn_kernels.cu
        ${CUDA_SOURCES}
    )
    
    target_link_libraries(fpn_tests
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
    )
    
    set_target_properties(fpn_tests PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "75;80;86;89"
    )
    
    add_test(NAME FPNKernelTests COMMAND fpn_tests)
endif()

# Python bindings
if(pybind11_FOUND)
    pybind11_add_module(unified_fpn_python
        python/fpn_pytorch.cpp
        ${CUDA_SOURCES}
    )
    
    target_link_libraries(unified_fpn_python PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
    )
    
    set_target_properties(unified_fpn_python PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "75;80;86;89"
    )
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/UnifiedFPNConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/UnifiedFPNConfig.cmake"
    INSTALL_DESTINATION lib/cmake/UnifiedFPN
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/UnifiedFPNConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/UnifiedFPNConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/UnifiedFPNConfigVersion.cmake"
    DESTINATION lib/cmake/UnifiedFPN
)

# Export targets
install(EXPORT UnifiedFPNTargets
    FILE UnifiedFPNTargets.cmake
    NAMESPACE UnifiedFPN::
    DESTINATION lib/cmake/UnifiedFPN
)

export(EXPORT UnifiedFPNTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/UnifiedFPNTargets.cmake"
    NAMESPACE UnifiedFPN::
)

# CPack configuration
set(CPACK_PACKAGE_NAME "UnifiedFPN")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance CUDA Feature Pyramid Network implementation")
set(CPACK_PACKAGE_VENDOR "Advanced CUDA Developers")
set(CPACK_PACKAGE_CONTACT "dev@example.com")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# Custom targets
add_custom_target(format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/python -name "*.cpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(lint
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include -name "*.cpp" -o -name "*.cu" -o -name "*.h" | xargs cppcheck --enable=all --std=c++17
    COMMENT "Running static analysis"
)

# Print configuration summary
message(STATUS "")
message(STATUS "UnifiedFPN Configuration Summary:")
message(STATUS "  Version: 1.0.0")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  Testing: ${GTest_FOUND}")
message(STATUS "  Python Bindings: ${pybind11_FOUND}")
message(STATUS "  Documentation: ${DOXYGEN_FOUND}")
message(STATUS "")