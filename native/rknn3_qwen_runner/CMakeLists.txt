cmake_minimum_required(VERSION 3.16)
project(rknn3_qwen_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rknn_qwen_runner src/main.cpp)
