
#include "test_util.h"
#include "errors.h"
#include <cstdlib>

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

  for (size_t i = 0; i < n_vals; ++i) {
    pos[i].row = static_cast<size_t>(std::rand()) % rows;
    pos[i].col = static_cast<size_t>(std::rand()) % cols;
    vals[i] = randf();
  }

  return Mat(rows, cols, pos, vals);
}

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}
// cudla::dense::Mat random_dense() {}
