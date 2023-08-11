
#include "test_util.h"
#include "errors.h"
#include <cstdlib>
#include <random>

// SMatF gen_random(long n_vals, long n_rows, long n_cols) {
//   float *vals = malloc(n_vals * sizeof(float));
//   long *pos_row = malloc(n_vals * sizeof(long));
//   long *pos_col = malloc(n_vals * sizeof(long));

//   gen_randoms(pos_row, pos_col, vals, n_vals, n_rows, n_cols);

//   SMatF ret = SM_from_pos_with(n_rows, n_cols, n_vals, pos_row, pos_col,
//   vals);

//   free(vals);
//   free(pos_row);
//   free(pos_col);

//   return ret;
// }

cudla::sparse::Mat random_sparse(size_t n_vals, size_t rows, size_t cols) {
  using namespace cudla::sparse;
  std::vector<float> vals(n_vals);
  std::vector<MPos> pos(n_vals);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (size_t i = 0; i < n_vals; ++i) {
    pos[i].row = static_cast<size_t>(std::rand()) % rows;
    pos[i].col = static_cast<size_t>(std::rand()) % cols;
    vals[i] = dis(gen);
  }

  return Mat(rows, cols, pos, vals);
}

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

cudla::dense::Mat random_dense(size_t rows, size_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  return {rows, cols, [&dis, &gen](size_t, size_t) { return dis(gen); }};
}

bool almost_equal(const cudla::dense::Mat &a, const cudla::dense::Mat &b,
                  float errtol) {
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
