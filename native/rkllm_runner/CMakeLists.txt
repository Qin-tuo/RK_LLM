cmake_minimum_required(VERSION 3.16)
project(rkllm_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rkllm_runner src/main.cpp)
