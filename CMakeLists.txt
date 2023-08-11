cmake_minimum_required(VERSION 3.21)

project(cudla)

set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(COMMON_CXXFLAGS
  -stdlib=libc++ -std=c++2b 
  -Wall
  -Werror
  -Wextra
  -Wconversion
  -Wunused
  -Wunused-value
  -Wunused-variable
  -Wno-string-plus-int
  )

add_library(cudla STATIC 
  ./src/dense.cpp
  ./src/sparse.cpp
  ./src/vec.cpp
)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  target_compile_definitions(cudla PRIVATE ASSERT_FILE_LOCS)
endif()



target_compile_options(
  cudla PRIVATE 
  ${COMMON_CXXFLAGS})

add_executable(test test/test.cpp test/test_util.cpp)

target_compile_options(
  test PRIVATE 
  ${COMMON_CXXFLAGS})

string(LENGTH "${CMAKE_SOURCE_DIR}/" SRC_PATH_SIZE)
target_compile_definitions(test PRIVATE SRC_PATH_SIZE=${SRC_PATH_SIZE})
target_include_directories(test PRIVATE ./src ./test)
target_link_libraries(test PRIVATE cudla)

target_link_options(test PRIVATE -std=c++2b -stdlib=libc++ -rdynamic)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_compile_options(cudla PRIVATE
  -fsanitize=undefined,address
)
target_link_options(cudla PRIVATE
  -fsanitize=undefined,address
)
target_link_options(test PRIVATE
  -fsanitize=undefined,address
)
endif()
