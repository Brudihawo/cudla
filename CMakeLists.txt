cmake_minimum_required(VERSION 3.21)

project(lacpp)

set(CMAKE_CXX_COMPILER /usr/bin/g++)

add_executable(test test/test.cpp)
target_compile_options(
  test
  PRIVATE -Wall
          -Werror
          -Wextra
          -Wconversion
          -Wno-string-plus-int
          -std=c++2b
          -rdynamic)

string(LENGTH "${CMAKE_SOURCE_DIR}/" SRC_PATH_SIZE)
target_compile_definitions(test PRIVATE SRC_PATH_SIZE=${SRC_PATH_SIZE})
target_include_directories(test PRIVATE ./src ./test)

target_link_options(test PRIVATE -std=c++2b -rdynamic)
