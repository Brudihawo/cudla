cmake_minimum_required(VERSION 3.21)

project(lacpp)

set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -Wconversion -std=c++2b")
# set(CMAKE_CXX_STANDARD c++2b)

add_executable(test test.cpp)

