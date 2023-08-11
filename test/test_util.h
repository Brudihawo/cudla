
#include "dense.h"
#include "sparse.h"

float randf();

cudla::dense::Mat random_dense();
cudla::sparse::Mat random_sparse(size_t n_vals, size_t rows, size_t cols);

