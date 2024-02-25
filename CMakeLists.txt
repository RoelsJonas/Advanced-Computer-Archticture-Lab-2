cmake_minimum_required(VERSION 3.21)
project(max_detection CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(max_detection main.cu)

set_target_properties(max_detection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
