cmake_minimum_required(VERSION 3.10)

project(demo LANGUAGES CXX CUDA)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
option(TEST OFF)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(cudalib SHARED cudalib.cu)
#target_link_libraries(cudalib nvinfer cudart)
#set_target_properties(cudalib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (TEST)
    message("testing ...")
    cuda_add_executable(${CMAKE_PROJECT_NAME} test.cpp mat.cu)
else()
    message("running ...")
    cuda_add_executable(${CMAKE_PROJECT_NAME} main.cpp mat.cu)
endif()
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${CMAKE_PROJECT_NAME} cudart)
target_link_libraries(${CMAKE_PROJECT_NAME} nvinfer)
#target_link_libraries(${CMAKE_PROJECT_NAME} cudalib)