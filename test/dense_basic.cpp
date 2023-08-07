#include "dense.h"
#include "test.h"

TEST_CASE(clone) {
  cudla::dense::Mat mat(ROWS, COLS);
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<float>(col + row * COLS);
    }
  }
  auto mat2 = mat.clone();

  TEST_ASSERT(mat2.operator==(mat));

  return TEST_SUCCESS;
}

TEST_CASE(print) {
  cudla::dense::Mat mat(ROWS, COLS);
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<float>(col + row * COLS);
    }
  }
  mat.print(out);
  return TEST_SUCCESS;
}

TEST_CASE(inv_mul_basic) {
  cudla::dense::Mat mat(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 2;
  }

  cudla::dense::Mat inv(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 0.5;
  }

  cudla::dense::Mat expected(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 1;
  }

  auto ret = mat * inv;
  std::stringstream s;
  s << "mat: \n" << mat << "\ninv: \n" << inv << "\nret" << ret;
  TEST_ASSERT_MSG(ret.operator==(expected), s.str());

  return TEST_SUCCESS;
}
