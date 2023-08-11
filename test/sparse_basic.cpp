#include "sparse.h"
#include "test.h"
#include "test_util.h"

using Mat = cudla::sparse::Mat;

#define SIZE 10
#define N_VALS 2

TEST_CASE(test_add_random) {
  Mat A = random_sparse(N_VALS, SIZE, SIZE);
  Mat B = random_sparse(N_VALS, SIZE, SIZE);

  Mat target = A + B;

  for (size_t row = 0; row < SIZE; ++row) {
    for (size_t col = 0; col < SIZE; ++col) {
      float actual = target[row, col];
      float expected = A[row, col] + B[row, col];

      TEST_ASSERT_MSG(actual == expected, "Matrix addition failed");
    }
  }
  return TEST_SUCCESS;
}

TEST_CASE(test_sub_random) {
  Mat A = random_sparse(N_VALS, SIZE, SIZE);
  Mat B = random_sparse(N_VALS, SIZE, SIZE);

  Mat target = A - B;

  for (size_t row = 0; row < SIZE; ++row) {
    for (size_t col = 0; col < SIZE; ++col) {
      float actual = target[row, col];
      float expected = A[row, col] - B[row, col];

      TEST_ASSERT_MSG(actual == expected, "Matrix addition failed");
    }
  }
  return TEST_SUCCESS;
}

TEST_CASE(test_scl_random) {
  Mat A = random_sparse(N_VALS, SIZE, SIZE);
  std::srand(69);
  float s = randf();

  Mat target = A * s;

  for (size_t row = 0; row < SIZE; ++row) {
    for (size_t col = 0; col < SIZE; ++col) {
      float t = target[row, col];
      float a = A[row, col];

      TEST_ASSERT_MSG(t == a * s, "Matrix addition failed");
    }
  }
  return TEST_SUCCESS;
}

