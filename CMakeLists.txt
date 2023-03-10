cmake_minimum_required(VERSION 3.21)

project(lacpp)

set(CMAKE_CXX_COMPILER /usr/bin/g++)

add_executable(test test.cpp)
target_compile_options(test PRIVATE -Wall -Werror -Wextra -Wconversion
                                    -std=c++2b)
