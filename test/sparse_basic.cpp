#include "sparse.h"
#include "test.h"
#include "test_util.h"

using Mat = cudla::sparse::Mat;

#define SIZE 8
#define N_VALS (5)

TEST_CASE(binary_search_random) {
  static constexpr size_t N = 4096;
  size_t *arr = new size_t[N];

  for (size_t i = 0; i < N; ++i) {
    arr[i] = static_cast<size_t>(std::rand() % RAND_MAX);
  }
  std::sort(arr, arr + N);

  for (size_t i = 0; i < 4096; ++i) {
    size_t rand_idx = static_cast<size_t>(std::rand()) % N;
    auto val = cudla::sparse::binary_search(arr, N, arr[rand_idx]);
    TEST_ASSERT(val.has_value() && val.value() == rand_idx);
  }

  delete[] arr;
  return TEST_SUCCESS;
}

TEST_CASE(binary_search_end) {
  static constexpr size_t N = 1024;
  size_t *arr = new size_t[N];

  for (size_t i = 0; i < N; ++i) {
    arr[i] = static_cast<size_t>(std::rand() % RAND_MAX);
  }
  std::sort(arr, arr + N);
  auto val = cudla::sparse::binary_search(arr, N, arr[N - 1]);
  TEST_ASSERT(val.has_value() && val.value() == N - 1);

  delete[] arr;
  return TEST_SUCCESS;
}

TEST_CASE(binary_search_begin) {
  static constexpr size_t N = 1024;
  size_t *arr = new size_t[N];

  for (size_t i = 0; i < N; ++i) {
    arr[i] = static_cast<size_t>(std::rand() % RAND_MAX);
  }
  std::sort(arr, arr + N);

  auto val = cudla::sparse::binary_search(arr, N, arr[0]);
  TEST_ASSERT(val.has_value() && val.value() == 0);

  delete[] arr;
  return TEST_SUCCESS;
}

TEST_CASE(initialization_from_pos) {
  using namespace cudla::sparse;
  std::vector<float> vals(N_VALS);
  std::vector<MPos> pos(N_VALS);

  for (size_t i = 0; i < N_VALS; ++i) {
    pos[i].row = static_cast<size_t>(std::rand()) % SIZE;
    pos[i].col = static_cast<size_t>(std::rand()) % SIZE;
    vals[i] = randf();
  }

  Mat mat(SIZE, SIZE, pos, vals);

  for (size_t i = 0; i < N_VALS; ++i) {
    size_t row = pos[i].row;
    size_t col = pos[i].col;
    float val = mat[row, col];
    TEST_ASSERT(val == vals[i]);
  }

  return TEST_SUCCESS;
}

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

TEST_CASE(test_transpose) {
  Mat A = random_sparse(N_VALS, SIZE, SIZE);
  Mat B = A.transposed();

  A.print_shape(std::cout);
  std::cout << "=========================\n";
  B.print_shape(std::cout);

  for (size_t col = 0; col < A.cols(); ++col) {
    for (size_t row = 0; row < A.rows(); ++row) {
      if (A.has_loc(row, col)) {
        TEST_ASSERT_MSG(B.has_loc(col, row), "structure transpose is correct");
      }
      float a = A[row, col];
      float b = B[col, row];
      TEST_ASSERT_MSG(a == b, "Matrix transpose failed");
    }
  }
  return TEST_SUCCESS;
}
