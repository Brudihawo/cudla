#include "sparse.h"
#include "errors.h"
#include <algorithm>
#include <optional>

namespace cudla::sparse {
std::optional<size_t> binary_search(const size_t *const arr, size_t size,
                                    size_t val) {
  if (size < 1)
    return std::nullopt;
  if (size == 1) {
    if (val != arr[0])
      return -1;
    else
      return 0;
  }

  size_t begin = 0;
  size_t end = size;
  size_t cur_sz = size;

  while (cur_sz > 0) {
    cur_sz = end - begin;
    const size_t mid_idx = begin + cur_sz / 2;
    const size_t mid_val = arr[mid_idx];

    if (val == mid_val)
      return mid_idx;

    const size_t ne = mid_idx - 1;
    const size_t nb = mid_idx + 1;
    const int cmp = val < mid_val;

    // hopefully this is faster?
    end = static_cast<size_t>(cmp) * ne + static_cast<size_t>(1 - cmp) * end;
    begin =
        static_cast<size_t>(cmp) * begin + static_cast<size_t>(1 - cmp) * nb;
    // if (val < mid_val) {
    //   end = mid_idx - 1;
    // } else {
    //   begin = mid_idx + 1;
    // }
  }

  return std::nullopt;
}

std::optional<size_t> Mat::col(size_t row, size_t col_idx) const {
  if (row >= rows_ || col_idx >= cols_ || row_sizes_[row] <= col_idx) {
    return std::nullopt;
  }

  return col_pos_[row_starts_[row] + col_idx];
}

Mat::Mat(size_t rows, size_t cols, size_t n_vals)
    : rows_(rows), cols_(cols), vals_(n_vals), col_sizes_(cols),
      col_starts_(cols), col_pos_(n_vals), row_starts_(rows), row_sizes_(rows) {
}

Mat Mat::clone_empty() const {
  Mat ret(rows_, cols_, vals_.size());

  std::copy(col_sizes_.begin(), col_sizes_.end(), ret.col_sizes_.data());
  std::copy(col_starts_.begin(), col_starts_.end(), ret.col_starts_.data());
  std::copy(col_pos_.begin(), col_pos_.end(), ret.col_pos_.data());
  std::copy(row_starts_.begin(), row_starts_.end(), ret.row_starts_.data());
  std::copy(row_sizes_.begin(), row_sizes_.end(), ret.row_sizes_.data());
  return ret;
}

Mat Mat::clone() const {
  Mat ret = this->clone_empty();
  std::copy(vals_.begin(), vals_.end(), ret.vals_.begin());
  return ret;
}

size_t Mat::rows() const { return rows_; }
size_t Mat::cols() const { return cols_; }

std::optional<size_t> Mat::idx(size_t row, size_t col) const {
  if (row >= rows_ || col >= rows_) {
    // TODO: re-enable logging
    // log_err("Position (%ld, %ld) out of bounds in Matrix of size (%ld,
    // %ld).",
    //         row, col, A.nrows, A.ncols);
    exit(EXIT_FAILURE);
  }

  if (col_sizes_[col] == 0)
    return std::nullopt;

  const size_t row_start = row_starts_[row];
  const std::optional<size_t> col_present =
      binary_search(&(col_pos_[row_start]), row_sizes_[row_start], col);
  return col_present.has_value()
             ? std::optional{row_start + col_present.value()}
             : std::nullopt;
}

/**
 * @brief return transposed version of self
 *
 * @return self transposed version of self
 */
Mat Mat::transposed() const {
  Mat ret(cols_, rows_, vals_.size());

  std::copy(row_starts_.begin(), row_starts_.end(), ret.col_starts_.data());
  std::copy(col_starts_.begin(), col_starts_.end(), ret.row_starts_.data());

  std::copy(col_sizes_.begin(), col_sizes_.end(), ret.row_sizes_.data());
  std::copy(row_sizes_.begin(), row_sizes_.end(), ret.col_sizes_.data());

  std::copy(col_pos_.begin(), col_pos_.end(), ret.col_pos_.data());

  std::vector<Pos> r_pos(vals_.size());
  size_t idx = 0;
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col_idx = 0; col_idx < row_sizes_[col_idx]; ++col_idx) {
      const size_t col = this->col(row, col_idx).value();
      r_pos[idx] = {row, col};
    }
  }

  std::sort(r_pos.begin(), r_pos.end(), [](const Pos &a, const Pos &b) -> bool {
    if (a.row == b.row) {
      return a.col > b.col;
    }
    return a.row > b.row;
  });

  size_t i = 0;
  for (const auto &pos : r_pos) {
    ret.col_pos_[i++] = pos.col;
    ret(pos.row, pos.col) = (*this)[pos.col, pos.row];
  }
  return ret;
}

// // TODO: implement transpose_in_place
// float SM_at(const SMatF A, long row, long col) {
//   if (row >= A.nrows || col >= A.ncols) {
//     log_err(
//         "Position (%ld, %ld) is out of bounds for matrix of size (%ld, %ld)",
//         row, col, A.nrows, A.ncols);
//     exit(EXIT_FAILURE);
//   }

//   long idx = SM_idx(A, row, col);
//   return idx == SM_NOT_PRESENT ? 0.0f : A.vals[idx];
// }

float Mat::operator[](size_t row, size_t col) const {
  const std::optional<size_t> index = idx(row, col);
  if (index.has_value()) {
    return vals_[index.value()];
  }
  return 0.0f;
}

float &Mat::operator()(size_t row, size_t col) {
  const std::optional<size_t> index = idx(row, col);
  cudla_assert_msg(index.has_value(),
                   "can only return refernce to non-zero value in matrix");
  return vals_[index.value()];
}

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

// TODO: we dont want this to exist
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

// this needs to work some other way
float Mat::det() const {
  auto tmp = this->make_triu();
  float ret = 0.0;
  for (size_t i = 0; i < rows_; ++i) {
    ret *= tmp[i, i];
  }
  return ret;
}

bool Mat::structure_eq(const Mat &other) const {
  // TODO: I think this can be optimized. I dont think we have to do all these
  // checks.
  if (this->cols_ != other.cols_ || this->rows_ != other.rows_ ||
      this->vals_ != other.vals_)
    return false;

  if (this->row_sizes_ != other.row_sizes_) {
    return false;
  }
  if (this->row_starts_ != other.row_starts_) {
    return false;
  }
  if (this->col_pos_ != other.col_pos_) {
    return false;
  }

  // // TODO: is this correct without these comparisons? I think so.
  // if (this->col_sizes_ != other.col_sizes_) {
  //   return false;
  // }
  // if (this->col_starts_ != other.col_starts_) {
  //   return false;
  // }

  return true;
}

bool Mat::has_loc(size_t row, size_t col) const {
  cudla_assert_msg((row >= this->rows_ || col >= this->cols_) ||
                       (row < 0 || col < 0),
                   "Position out of bounds");

  // Check if row is present in matrix
  if (this->row_empty(row)) {
    return false;
  }

  const size_t row_start = this->row_starts_[row];
  const std::optional<size_t> pos =
      binary_search(&this->col_pos_[row_start], this->row_sizes_[row], col);

  if (pos.has_value()) {
    return true;
  }
  return false;
}

void Mat::init_start_arrs() {
  this->row_starts_[0] = 0;
  for (size_t row = 1; row < this->rows_; ++row) {
    this->row_starts_[row] =
        this->row_starts_[row - 1] + this->row_sizes_[row - 1];
  }

  this->col_starts_[0] = 0;
  for (size_t col = 1; col < this->cols_; ++col) {
    this->col_starts_[col] =
        this->col_starts_[col - 1] + this->col_sizes_[col - 1];
  }
}

Mat Mat::addsub_alloc(const Mat &other) const {
  cudla_assert_msg(
      this->rows_ == other.rows_ && this->cols_ == other.cols_,
      "Size mismatch: this and other need to have the same dimensions.");
  if (this->structure_eq(other)) {
    return this->clone_empty();
  }

  size_t vals = 0;
  for (size_t col = 0; col < other.cols_; ++col) {
    for (size_t row = 0; row < this->rows_; ++row) {
      if (this->has_loc(row, col) || other.has_loc(row, col)) {
        ++vals;
      }
    }
  }

  // TODO: Optimize
  Mat ret(this->rows_, this->cols_, vals);
  size_t cur_idx = 0;
  for (size_t row = 0; row < this->rows_; ++row) {
    for (size_t col = 0; col < this->cols_; ++col) {
      if (this->has_loc(row, col) || other.has_loc(row, col)) {
        ret.col_pos_[cur_idx] = col;
        ++ret.row_sizes_[row];
        ++ret.col_sizes_[col];
        ++cur_idx;
      }
    }
  }

  ret.init_start_arrs();
  return ret;
}

Mat Mat::operator+(const Mat &o) {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix addition");
  Mat ret = this->addsub_alloc(o);
  for (size_t row = 0; row < ret.rows_; ++row) {
    for (size_t col_idx = 0; col_idx < ret.row_sizes_[row]; ++col_idx) {
      const std::optional<size_t> maybe_col = ret.col(row, col_idx);
      cudla_assert_msg(maybe_col.has_value(),
                       "tried to access column in matric addition");
      const size_t col = maybe_col.value();
      ret(row, col) = this->operator[](row, col) + o[row, col];
    }
  }
  return ret;
}

Mat Mat::operator-(const Mat &o) {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix subtraction");
  Mat ret = this->addsub_alloc(o);
  for (size_t row = 0; row < ret.rows_; ++row) {
    for (size_t col_idx = 0; col_idx < ret.row_sizes_[row]; ++col_idx) {
      const std::optional<size_t> maybe_col = ret.col(row, col_idx);
      cudla_assert_msg(maybe_col.has_value(),
                       "tried to access column in matric addition");
      const size_t col = maybe_col.value();
      ret(row, col) = this->operator[](row, col) - o[row, col];
    }
  }
  return ret;
}

void Mat::operator*=(float r) {
  for (size_t i = 0; i < this->vals_.size(); ++i) {
    this->vals_[i] *= r;
  }
}

// size_t rows_ = 0, cols_ = 0;
// std::vector<float> vals_;

// std::vector<size_t> col_sizes_;
// std::vector<size_t> col_starts_;
// std::vector<size_t> col_idcs_;
// std::vector<size_t> col_pos_;
// std::vector<size_t> row_starts_;
// std::vector<size_t> row_sizes_;

Mat::Mat(size_t nvals, std::vector<size_t> row_sizes,
         std::vector<size_t> col_sizes, std::vector<size_t> row_starts,
         std::vector<size_t> col_starts, std::vector<size_t> col_pos)
    : rows_(row_sizes.size()), cols_(col_sizes.size()), vals_(nvals),
      col_starts_(std::move(col_starts)), col_pos_(std::move(col_pos)),
      row_starts_(std::move(row_starts)) {}

Mat Mat::prod_alloc(const Mat &other) const {
  cudla_assert_msg(this->cols_ == other.rows_,
                   "Size mismatch, needs this->cols_ == other.rows_.");

  size_t tmp_buf_init_cap = (this->vals_.size() + other.vals_.size()) * 2;

  // Mat ret = {
  //     .nrows = this->rows_,
  //     .ncols = other.cols_,
  //     .nvals = 0, // tbd

  //     .col_sizes = calloc(other.cols_, sizeof(long)),
  //     .col_starts = calloc(other.cols_, sizeof(long)),
  //     .col_pos = NULL,
  //     .vals = NULL,
  //     .row_starts = calloc(A.nrows, sizeof(long)),
  //     .row_sizes = calloc(A.nrows, sizeof(long)),
  // };

  // ret.col_sizes_.reserve(other.cols_);
  // ret.col_sizes_.reserve(other.cols_);

  size_t rows = this->rows_;
  size_t cols = other.cols_;
  size_t nvals = 0;
  std::vector<size_t> row_sizes(rows, 0);
  std::vector<size_t> row_starts(rows, 0);
  std::vector<size_t> col_sizes(cols, 0);
  std::vector<size_t> col_starts(cols, 0);
  std::vector<size_t> col_pos;
  col_pos.reserve(tmp_buf_init_cap);

  // get memory requirements
  for (size_t t_row = 0; t_row < rows; t_row++) {
    for (size_t test_idx = 0; test_idx < this->row_sizes_[t_row]; test_idx++) {
      const std::optional<size_t> maybe_a_col = this->col(t_row, test_idx);
      // TODO: Better error message
      cudla_assert_msg(maybe_a_col.has_value(), "column needs to exist");
      const size_t a_col = maybe_a_col.value();
      // we get a column index in a and iterate over the corresponding column in
      // B then, we set all increment all appropriate sizes this will cause more
      // values per row than we actually have
      for (size_t t_col_i = 0; t_col_i < other.row_sizes_[a_col]; t_col_i++) {
        const std::optional<size_t> maybe_t_col = this->col(t_row, test_idx);
        // TODO: Better error message
        cudla_assert_msg(maybe_t_col.has_value(), "column needs to exist");
        const size_t t_col = maybe_t_col.value();

        ++nvals;
        ++col_sizes[t_col];
        ++row_sizes[t_row];
        col_pos.push_back(t_col);
      }
    }
  }

  // Set row / colum starts with wrong row sizes so we can iterate more easily
  size_t max_row_size = row_sizes[0];
  for (size_t row = 1; row < rows; ++row) {
    row_starts[row] = row_starts[row - 1] + row_sizes[row - 1];
    if (max_row_size < row_sizes[row])
      max_row_size = row_sizes[row];
  }

  // #define RADIX
  // #ifdef RADIX
  //   long *tmp_arr = malloc(max_row_size * sizeof(long));
  // #endif

  // remove duplicate columns that we got from position collection earlier
  // sort row-subsets of col_pos
  for (size_t row = 0; row < rows; ++row) {
    if (row_sizes[row] > 1) {
#ifdef RADIX
      radix_sort(&col_pos.vals[ret.row_starts[row]], ret.row_sizes[row],
                 tmp_arr);
#else
      std::sort(col_pos.begin() + static_cast<long>(row_starts[row]),
                col_pos.begin() +
                    static_cast<long>(row_starts[row] + row_sizes[row]));
#endif
    }
  }

  // #ifdef RADIX
  //   free(tmp_arr);
  // #endif
  //
#define NOT_PRESENT SIZE_MAX

  // find repeated values in col_pos and remove them
  for (size_t row = 0; row < rows; ++row) {
    if (row_sizes[row] > 1) {
      const size_t cur_row_size = row_sizes[row];
      size_t to_remove = 0;
      for (size_t i = row_starts[row] + 1; i < row_starts[row] + cur_row_size;
           ++i) {
        if (col_pos[i] == col_pos[i - 1]) {
          ++to_remove;
          col_pos[i - 1] = NOT_PRESENT;
          --col_sizes[col_pos[i]];
        }
      }

      row_sizes[row] -= to_remove;
      nvals -= to_remove;
    }
  }

  std::vector<size_t> ret_col_pos(nvals);

  size_t count = 0;
  for (size_t idx = 0; idx < col_pos.size(); ++idx) {
    if (ret_col_pos[idx] != NOT_PRESENT) {
      col_pos[count] = col_pos[idx];
      ++count;
    }
  }

  // Mat ret = {
  //     .nrows = this->rows_,
  //     .ncols = other.cols_,
  //     .nvals = 0, // tbd

  //     .col_sizes = calloc(other.cols_, sizeof(long)),
  //     .col_starts = calloc(other.cols_, sizeof(long)),
  //     .col_pos = NULL,
  //     .vals = NULL,
  //     .row_starts = calloc(A.nrows, sizeof(long)),
  //     .row_sizes = calloc(A.nrows, sizeof(long)),
  // };

  Mat ret(nvals, col_sizes, col_starts, row_sizes, row_starts, ret_col_pos);

  ret.init_start_arrs();

  return ret;
}

Mat Mat::operator*(const Mat &o) {
  cudla_assert_msg(this->cols_ == o.rows_,
                   "Size mismatch in matrix multiplication");
  Mat ret = this->prod_alloc(o);
  std::fill(ret.vals_.begin(), ret.vals_.end(), 0.0f);

  for (size_t t_row = 0; t_row < ret.rows_; t_row++) { // rows in target
    for (size_t a_col_i = 0; a_col_i < this->row_sizes_[t_row]; a_col_i++) {
      // iterate over columns present in row of A
      // idx is column in a / row in b
      const std::optional<size_t> maybe_col_idx = this->col(t_row, a_col_i);
      cudla_assert_msg(maybe_col_idx.has_value(), "column needs to be present");
      const size_t idx = maybe_col_idx.value();

      const size_t a_idx = this->row_starts_[t_row] + a_col_i;

      for (size_t t_col_i = 0; t_col_i < o.row_sizes_[idx]; t_col_i++) {
        const std::optional<size_t> maybe_t_col = o.col(idx, t_col_i);
        cudla_assert_msg(maybe_t_col.has_value(), "column needs to be present");
        const size_t t_col = maybe_t_col.value(); // column in target
        const size_t b_idx = o.row_starts_[idx] + t_col_i;

        ret(t_row, t_col) += this->vals_[a_idx] * o.vals_[b_idx];
      }
    }
  }
  return ret;
}

Mat<T> back_sub(const Mat<T> &b) const {
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
Mat<T> gauss_elim_mut(const Mat<T> &b) {
  cudla_assert_msg(this->rows_ == this->cols_, "Matrix needs to be square");
  cudla_assert_msg(b.cols_ == 1, "Target needs to be a vector");
  cudla_assert_msg(b.rows_ == this->rows_,
                   "Target rows need to match matrix dimensions");

  std::vector<T> cpybuf(cols_);
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
        double tmp = b2[k, 0];
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

std::optional<util::RowPerms> cholesky_decomp_mut() {
  util::RowPerms perms(rows_);
  std::vector<T> cpybuf(cols_);

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

void row_permute(const util::RowPerms &perms) {
  if (perms.n_swaps_ == 0)
    return;

  std::vector<T> cpybuf(cols_);

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

Mat<T> lu_forw_sub(const Mat<T> &b) {
  cudla_assert_msg(rows_ == cols_, "Matrix needs to be square");
  // TODO: assert tri_lo
  cudla_assert_msg(cols_ == b.rows(), "Vector rows need to match matrix cols");

  Mat<T> t(rows_, 1);
  t(0, 0) = b[0, 0];
  for (size_t i = 1; i < b.rows(); i++) {
    t(i, 0) = b[i, 0];
    for (size_t j = 0; j < i; j++) {
      t(i, 0) += t(j, 0) * (*this)[i, j];
    }
  }

  return t;
}

Mat<T> cholesky_decomp_solv(const Mat<T> &b) {
  cudla_assert_msg(this->rows_ == this->cols_, "Matrix needs to be square");
  cudla_assert_msg(b.cols_ == 1, "Target needs to be a vector");
  cudla_assert_msg(b.rows_ == this->rows_,
                   "Target rows need to match matrix dimensions");

  // TODO: This is broken
  std::vector<T> cpybuf(cols_);
  auto b2 = b.clone();

  const auto perms = this->cholesky_decomp_mut();
  if (perms.has_value()) {
    b2.row_permute(perms.value());
  }

  auto y = this->lu_forw_sub(b);

  return this->back_sub(y);
}

void print(auto &out) const {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      const T val = (*this)[row, col];
      out << std::setw(8) << std::setprecision(3) << val;
      if (col < cols_ - 1) {
        out << ", ";
      }
    }
    out << "\n";
  }
}

friend std::ostream &operator<<(std::ostream &stream, const Mat<T> &mat) {
  mat.print(stream);
  return stream;
}
}
} // cudla::sparse
