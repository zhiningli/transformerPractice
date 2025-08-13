cmake_minimum_required(VERSION 3.16)
project(TransformerPractice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Find required packages
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

# Main executable
add_executable(transformer_demo examples/main.cpp)
target_link_libraries(transformer_demo transformer_lib)

# Set output directories
set_target_properties(transformer_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
) 
