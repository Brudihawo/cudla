
#include "dense.h"
#include "sparse.h"

float randf();
cudla::sparse::Mat random_sparse(size_t n_vals, size_t rows, size_t cols);
cudla::dense::Mat random_dense(size_t rows, size_t cols);
bool almost_equal(const cudla::dense::Mat &a, const cudla::dense::Mat &b,
                  float errtol);
