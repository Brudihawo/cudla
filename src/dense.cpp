#include "dense.h"
#include "errors.h"
#include "vec.h"
#include <cstddef>
#include <vector>

namespace cudla::dense {

size_t Mat::idx(size_t row, size_t col) const { return row * cols_ + col; }

void Mat::swap_rows(size_t i, size_t j, std::vector<float> &cpybuf) {
  cudla_assert_msg(i < rows_, "Row out of bounds");
  cudla_assert_msg(j < rows_, "Row out of bounds");
  cudla_assert_msg(i != j, "Cannot swap row with itself");
  std::copy(vals_ + static_cast<long>(i * cols_),
            vals_ + static_cast<long>((i + 1) * cols_), cpybuf.begin());
  std::copy(vals_ + static_cast<long>(j * cols_),
            vals_ + static_cast<long>((j + 1) * cols_),
            vals_ + static_cast<long>(i * cols_));
  std::copy(cpybuf.begin(), cpybuf.end(), vals_ + static_cast<long>(j * cols_));
}

Mat::Mat(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), vals_(new float[rows_ * cols_]) {
  std::fill(vals_, vals_ + (rows_ * cols_), 0.0f);
}

Mat::Mat(size_t rows, size_t cols, float val)
    : rows_(rows), cols_(cols), vals_(new float[rows_ * cols_]) {

  for (size_t i = 0; i < rows_ * cols_; ++i) {
    vals_[i] = val;
  }
}

/**
 * @brief create a matrix and populate entries using a predicate
 *
 * @param rows number of rows in matrix
 * @param cols number of columns in matrix
 * @param pred predicate (row, column) -> value for assigning elements
 */
Mat::Mat(size_t rows, size_t cols, std::function<float(size_t, size_t)> pred)
    : rows_(rows), cols_(cols), vals_(new float[rows_ * cols_]) {

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      vals_[row * cols_ + col] = pred(row, col);
    }
  }
}

Mat::~Mat() {
  delete[] vals_;
}

Mat Mat::clone_empty() const { return Mat(rows_, cols_); }
Mat Mat::clone() const {
  Mat ret(rows_, cols_);
  std::copy(vals_, vals_ + (rows_ * cols_), ret.vals_);
  return ret;
}

size_t Mat::rows() const { return rows_; }
size_t Mat::cols() const { return cols_; }

/**
 * @brief return transposed version of self
 *
 * @return self transposed version of self
 */
Mat Mat::transposed() const {
  Mat ret(cols_, rows_);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      ret.vals_[ret.idx(col, row)] = vals_[idx(row, col)];
    }
  }
  return ret;
}

// TODO: implement transpose_in_place

float Mat::operator[](size_t row, size_t col) const {
  return vals_[idx(row, col)];
}
float &Mat::operator()(size_t row, size_t col) { return vals_[idx(row, col)]; }

bool Mat::operator==(const Mat &o) {
  if (o.rows_ != rows_ || o.cols_ != cols_) {
    return false;
  }

  for (size_t i = 0; i < rows_ * cols_; ++i) {
    if (o.vals_[i] != vals_[i]) {
      return false;
    }
  }
  return true;
}

Mat Mat::make_triu() const {
  cudla_assert_msg(rows_ == cols_, "matrix needs to be square");
  auto ret = this->clone();

  for (size_t zero_col = 0; zero_col < cols_ - 1;
       zero_col++) { // zeroing of i-th column
    for (size_t row = zero_col + 1; row < rows_; row++) {   // row n
      for (size_t col = zero_col + 1; col < cols_; col++) { // column j
        ret(row, col) -=
            ret[row, zero_col] / ret[zero_col, zero_col] * ret[zero_col, col];
      }
    }
  }

  for (size_t i = 1; i < rows_; i++) {
    for (size_t j = 0; j < i; j++) {
      ret(i, j) = 0.0;
    }
  }
  return ret;
}

float Mat::det() const {
  auto tmp = this->make_triu();
  float ret = 0.0;
  for (size_t i = 0; i < rows_; ++i) {
    ret *= tmp[i, i];
  }
  return ret;
}

Mat Mat::operator+(const Mat &o) {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix subtraction");
  Mat ret(rows_, cols_);
  for (size_t i = 0; i < rows_ * cols_; ++i) {
    ret.vals_[i] = this->vals_[i] + o.vals_[i];
  }
  return ret;
}

Mat Mat::operator-(const Mat &o) {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix subtraction");
  Mat ret(rows_, cols_);
  for (size_t i = 0; i < rows_ * cols_; ++i) {
    ret.vals_[i] = this->vals_[i] - o.vals_[i];
  }
  return ret;
}

Mat Mat::operator*(const Mat &o) {
  cudla_assert_msg(cols_ == o.rows_, "Size mismatch in matrix multiplication");

  Mat ret(rows_, o.cols_);

  for (size_t col = 0; col < ret.cols_; col++) {
    for (size_t row = 0; row < ret.rows_; row++) {
      ret(row, col) = 0.0;
      for (size_t idx = 0; idx < o.rows_; idx++) {
        ret(row, col) += (*this)[row, idx] * o[idx, col];
      }
    }
  }
  return ret;
}

Mat Mat::back_sub(const Mat &b) const {
  cudla_assert_msg(rows_ == cols_, "this needs to be a square matrix");
  cudla_assert_msg(b.cols() == 1, "b needs to be a column vector");
  cudla_assert_msg(b.rows() == cols_, "size mismatch");
  // TODO: somehow assert (if I want that) that this mat is tri-up

  auto t = b.clone_empty();

  t(t.rows() - 1, 0) = b[b.rows() - 1, 0] / (*this)[rows_ - 1, cols_ - 1];

  for (size_t i = b.rows() - 2; i + 1 > 0; --i) { // i + 1 > 0 to stop underflow
    t(i, 0) = b[i, 0];
    for (size_t j = i + 1; j < b.rows(); ++j) {
      t(i, 0) -= t[j, 0] * (*this)[i, j];
    }
    t(i, 0) /= (*this)[i, i];
  }
  return t;
}

/**
 * @brief solve the system of equations this * x = b
 * this needs to be a square matrix and target needs to be a column vector
 * this matrix is modified to a triangular upper matrix
 *
 * @param target right hand side of the system of equations
 * @return x - solution to system of equations
 */
Mat Mat::gauss_elim_mut(const Mat &b) {
  cudla_assert_msg(this->rows_ == this->cols_, "Matrix needs to be square");
  cudla_assert_msg(b.cols_ == 1, "Target needs to be a vector");
  cudla_assert_msg(b.rows_ == this->rows_,
                   "Target rows need to match matrix dimensions");

  std::vector<float> cpybuf(cols_);
  util::RowPerms perms(rows_);
  auto b2 = b.clone();

  for (size_t i = 0; i < cols_ - 1; i++) {   // zeroing of i-th column
    for (size_t n = i + 1; n < rows_; n++) { // row n
      if ((*this)[i, i] == 0.0) {            // pivot
        size_t k = 0;
        for (; (*this)[k, i] == 0.0; k++)
          ;
        // swap rows of a
        this->swap_rows(i, k, cpybuf);

        // swap values in b2
        float tmp = b2[k, 0];
        b2(k, 0) = b2[i, 0];
        b2(i, 0) = tmp;

        // increase permutation count and record permutation
        perms.add_perm(i, k);
      }
      for (size_t j = i + 1; j < cols_; j++) { // column j
        (*this)(n, j) -= (*this)[n, i] / (*this)[i, i] * (*this)[i, j];
      }
      b2(n, 0) -= (*this)[n, i] / (*this)[i, i] * b2[i, 0];
      (*this)(n, i) = 0.0;
    }
  }

  // invertibility check
  // TODO: check for invertibility of A
  for (size_t i = 0; i < cols_; i++) {
    cudla_assert_msg(
        ((*this)[i, i] != 0.0),
        "Diagonal of triangular upper version contains zero-element");
  }

  auto ret = this->back_sub(b2);
  // for (size_t i = 0; i < cols_; i++) {
  //   std::cout << perms.order_[i] << ", ";
  // }
  // std::cout << "\n";

  // TODO: i need to do something with the permutations here, but for some
  // reason it seems to be correct???? Why?
  return ret;
}

std::optional<util::RowPerms> Mat::cholesky_decomp_mut() {
  util::RowPerms perms{rows_};
  std::vector<float> cpybuf(cols_);

  for (size_t i = 0; i < cols_ - 1; i++) {     // zeroing of i-th column
    for (size_t n = i + 1; n < rows_; n++) {   // row n
      for (size_t j = i + 1; j < cols_; j++) { // column j
        if ((*this)[i, i] == 0.0) {            // pivot
          size_t k = 0;
          for (; (*this)[k, i] == 0.0; k++)
            ;
          // swap rows of a
          this->swap_rows(i, k, cpybuf);

          // increase permutation count and record permutation
          perms.add_perm(i, k);
        }
        (*this)(n, j) -= (*this)[n, i] / (*this)[i, i] * (*this)[i, j];
      }
      (*this)(n, i) = -(*this)[n, i] / (*this)[i, i];
    }
  }

  if (perms.n_swaps_ == 0) {
    return std::nullopt;
  }

  return perms;
}

void Mat::row_permute(const util::RowPerms &perms) {
  if (perms.n_swaps_ == 0)
    return;

  std::vector<float> cpybuf(cols_);

  size_t performed_perms = 0;
  for (size_t row = 0; row < rows_; row++) {
    if (performed_perms == perms.n_swaps_)
      break;

    if (perms.order_[row] != row) {
      this->swap_rows(row, perms.order_[row], cpybuf);
      performed_perms++;
    }
  }
}

Mat Mat::lu_forw_sub(const Mat &b) {
  cudla_assert_msg(rows_ == cols_, "Matrix needs to be square");
  // TODO: assert tri_lo
  cudla_assert_msg(cols_ == b.rows(), "Vector rows need to match matrix cols");

  Mat t(rows_, 1);
  t(0, 0) = b[0, 0];
  for (size_t i = 1; i < b.rows(); i++) {
    t(i, 0) = b[i, 0];
    for (size_t j = 0; j < i; j++) {
      t(i, 0) += t(j, 0) * (*this)[i, j];
    }
  }

  return t;
}

Mat Mat::cholesky_decomp_solv(const Mat &b) {
  cudla_assert_msg(this->rows_ == this->cols_, "Matrix needs to be square");
  cudla_assert_msg(b.cols_ == 1, "Target needs to be a vector");
  cudla_assert_msg(b.rows_ == this->rows_,
                   "Target rows need to match matrix dimensions");

  // TODO: This is broken
  std::vector<float> cpybuf(cols_);
  auto b2 = b.clone();

  const auto perms = this->cholesky_decomp_mut();
  if (perms.has_value()) {
    b2.row_permute(perms.value());
  }

  auto y = this->lu_forw_sub(b);

  return this->back_sub(y);
}

cudla::dense::Vec operator*(const Mat &mat, const cudla::dense::Vec &vec) {
  cudla_assert_msg(
      mat.cols_ == vec.rows,
      "Matrix columns need to match vector rows for multiplication");

  Vec ret(vec.rows, 0.0f);
  for (size_t row = 0; row < vec.rows; ++row) {
    for (size_t col = 0; col < mat.cols_; ++col) {
      ret(row) += mat[row, col] * vec[row];
    }
  }

  return ret;
}

std::ostream &operator<<(std::ostream &stream, const Mat &mat) {
  mat.print(stream);
  return stream;
}

} // namespace cudla::dense
