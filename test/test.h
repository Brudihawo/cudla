#pragma once
#include "dense.h"
#include <iostream>
#include <random>
#include <sstream>
#include <string.h>
#include <vector>

using test_fun = bool (*)(std::stringstream &cout, std::stringstream &cerr);

struct Test {
  const char *name;
  const char *file;
  const size_t line;
  test_fun fun;
};

struct TestResult {
  const char *name;
  const char *file;
  const size_t line;
  bool success_;
  std::string out_;
  std::string err_;

  TestResult(const Test &test)
      : name(test.name), file(test.file), line(test.line) {
    std::stringstream out;
    std::stringstream err;
    success_ = test.fun(out, err);
    out_ = out.str();
    err_ = err.str();
  }
};

inline cudla::dense::Mat<double> random_mat(size_t rows, size_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  return {rows, cols, [&dis, &gen](size_t, size_t) { return dis(gen); }};
}

template <typename T>
bool almost_equal(const cudla::dense::Mat<T> &a, const cudla::dense::Mat<T> &b,
                  T errtol) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }

  for (size_t row = 0; row < b.rows(); ++row) {
    for (size_t col = 0; col < b.cols(); ++col) {
      if (std::abs(a[row, col] - b[row, col]) > errtol) {
        return false;
      }
    }
  }
  return true;
}

#define ROWS 15
#define COLS 15

#define ERRTOL 0.0001

#define RED "\033[31;1m"
#define GRN "\033[32;1m"
#define RST "\033[0;0m"

#define TEST_SUCCESS true
#define TEST_FAILURE false

#define __RELPATH__ (__FILE__ + SRC_PATH_SIZE)

#define ADD_FUNC(func, file_, line_)                                           \
  static Test ptr_##func __attribute((used, section("test_functions"))) = {    \
      .name = #func, .file = file_, .line = line_, .fun = func}

#define section_foreach_entry(section_name, type_t, elem)                      \
  for (type_t *elem = ({                                                       \
         extern type_t __start_##section_name;                                 \
         &__start_##section_name;                                              \
       });                                                                     \
       elem != ({                                                              \
         extern type_t __stop_##section_name;                                  \
         &__stop_##section_name;                                               \
       });                                                                     \
       ++elem)

#define TEST_CASE(name) TEST_CASE_(name, __RELPATH__, __LINE__)

#define TEST_CASE_(name, file, line)                                           \
  bool(name)(std::stringstream & out, std::stringstream & err);                \
  ADD_FUNC(name, file, line);                                                  \
  bool(name)([[maybe_unused]] std::stringstream & out,                         \
             [[maybe_unused]] std::stringstream & err)

#define TEST_ASSERT(cond) TEST_ASSERT_(cond, __RELPATH__, __LINE__)
#define TEST_ASSERT_(cond, file, line)                                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      err << file << ":" << line << ": TEST_ASSERT(" #cond ") failed";         \
      return TEST_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define TEST_ASSERT_MSG(cond, msg)                                             \
  TEST_ASSERT_MSG_(cond, msg, __RELPATH__, __LINE__)
#define TEST_ASSERT_MSG_(cond, msg, file, line)                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      err << file << ":" << line << ": TEST_ASSERT(" #cond ") failed\n";       \
      err << msg;                                                              \
      return TEST_FAILURE;                                                     \
    }                                                                          \
  } while (0)
