#include "sparse.h"
#include "errors.h"
#include <algorithm>
#include <optional>

namespace cudla::sparse {
std::optional<size_t> binary_search(const size_t *const arr, size_t size,
                                    size_t val) {
  if (size == 0)
    return std::nullopt;
  if (size == 1)
    return arr[0] == val ? std::optional<size_t>(0) : std::nullopt;

  size_t l = 0;
  size_t r = size - 1;

  while (l <= r) {
    const size_t mid_idx = (l + r) / 2;
    const size_t mid_val = arr[mid_idx];
    if (mid_val < val) {
      l = mid_idx + 1;
    } else if (mid_val > val) {
      if (mid_idx == 0)
        return std::nullopt;
      r = mid_idx - 1;
    } else {
      return mid_idx;
    }
  }

  return std::nullopt;
}

size_t Mat::col(size_t row, size_t col_idx) const {
  if (row >= rows_ || col_idx >= row_sizes_[row]) {
    return NOT_PRESENT;
  }

  return col_pos_[row_starts_[row] + col_idx];
}

size_t Mat::col_or_panic(size_t row, size_t col_idx) const {
  cudla_assert(row < rows_ || col_idx < row_sizes_[row]);

  return col_pos_[row_starts_[row] + col_idx];
}

Mat::Mat(size_t rows, size_t cols, size_t n_vals)
    : rows_(rows), cols_(cols), n_vals_(n_vals), vals_(new float[n_vals]),
      col_pos_(new size_t[n_vals]), col_sizes_(new size_t[cols]),
      col_starts_(new size_t[cols]), row_starts_(new size_t[rows]),
      row_sizes_(new size_t[rows]) {}

Mat Mat::clone_empty() const {
  Mat ret(rows_, cols_, n_vals_);

  std::copy(col_pos_, col_pos_ + n_vals_, ret.col_pos_);
  std::copy(col_sizes_, col_sizes_ + cols_, ret.col_sizes_);
  std::copy(col_starts_, col_starts_ + cols_, ret.col_starts_);
  std::copy(row_starts_, row_starts_ + rows_, ret.row_starts_);
  std::copy(row_sizes_, row_sizes_ + rows_, ret.row_sizes_);
  return ret;
}

Mat Mat::clone() const {
  Mat ret = this->clone_empty();
  std::copy(vals_, vals_ + n_vals_, ret.vals_);
  return ret;
}

size_t Mat::rows() const { return rows_; }
size_t Mat::cols() const { return cols_; }

size_t Mat::idx(size_t row, size_t col) const {
  cudla_assert_msg((row <= rows_ || col <= rows_), "position out of bounds");

  if (col_sizes_[col] == 0)
    return NOT_PRESENT;

  const size_t row_start = row_starts_[row];
  const std::optional<size_t> col_present =
      binary_search(&(col_pos_[row_start]), row_sizes_[row], col);
  return col_present.has_value() ? row_start + col_present.value()
                                 : NOT_PRESENT;
}

/**
 * @brief return transposed version of self
 */
Mat Mat::transposed() const {
  Mat ret(cols_, rows_, n_vals_);

  std::copy(row_starts_, row_starts_ + rows_, ret.col_starts_);
  std::copy(col_starts_, col_starts_ + cols_, ret.row_starts_);

  std::copy(col_sizes_, col_sizes_ + cols_, ret.row_sizes_);
  std::copy(row_sizes_, row_sizes_ + rows_, ret.col_sizes_);

  std::vector<MPos> r_pos(n_vals_);
  size_t idx = 0;
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col_idx = 0; col_idx < row_sizes_[row]; ++col_idx) {
      const size_t col = this->col_or_panic(row, col_idx);
      r_pos[idx] = {col, row};
      ++idx;
    }
  }

  std::sort(r_pos.begin(), r_pos.end(),
            [](const MPos &l, const MPos &r) -> bool {
              if (l.row == r.row) {
                return l.col < r.col;
              }
              return l.row < r.row;
            });

  for (size_t i = 0; i < n_vals_; ++i) {
    ret.col_pos_[i] = r_pos[i].col;
  }

  for (const auto &pos : r_pos) {
    ret(pos.row, pos.col) = (*this)[pos.col, pos.row];
  }
  return ret;
}

// TODO: implement transpose_in_place

float Mat::operator[](size_t row, size_t col) const {
  const size_t index = idx(row, col);
  if (index != NOT_PRESENT) {
    return vals_[index];
  }
  return 0.0f;
}

float &Mat::operator()(size_t row, size_t col) {
  size_t index = idx(row, col);
  cudla_assert_msg(index != NOT_PRESENT,
                   "can only return refernce to non-zero value in matrix. ",
                   "Position (", row, ", ", col, ") is not filled.");
  return vals_[index];
}

bool Mat::operator==(const Mat &o) const {
  // compare structure / short circuit
  if (!this->structure_eq(o))
    return false;

  // compare values
  if (std::memcmp(vals_, o.vals_, n_vals_ * sizeof(float)) != 0)
    return false;

  // if non-zero structure and values are equal, A and B are equal
  return true;
}

bool Mat::structure_eq(const Mat &other) const {
  // TODO: I think this can be optimized. I dont think we have to do all these
  // checks.
  if (this->cols_ != other.cols_ || 
      this->rows_ != other.rows_ ||
      this->n_vals_ != other.n_vals_)
    return false;

  if (std::memcmp(this->row_sizes_, other.row_sizes_,
                  this->rows_ * sizeof(size_t)) != 0) {
    return false;
  }
  if (std::memcmp(this->row_starts_, other.row_starts_,
                  this->rows_ * sizeof(size_t)) != 0) {
    return false;
  }
  if (std::memcmp(this->col_pos_, other.col_pos_,
                  this->n_vals_ * sizeof(size_t)) != 0) {
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
  cudla_assert_msg((row < this->rows_ || col < this->cols_),
                   "Position out of bounds");

  // Check if row is present in matrix
  if (row_sizes_[row] == 0) {
    return false;
  }

  const size_t row_start = this->row_starts_[row];
  const std::optional<size_t> pos =
      binary_search(&this->col_pos_[row_start], this->row_sizes_[row], col);

  return pos.has_value();
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
  for (size_t row = 0; row < this->rows_; ++row) {
    for (size_t col = 0; col < other.cols_; ++col) {
      if (this->has_loc(row, col) || other.has_loc(row, col)) {
        ++vals;
      }
    }
  }

  // TODO: Optimize
  Mat ret(this->rows_, this->cols_, vals);
  std::fill(ret.row_sizes_, ret.row_sizes_ + ret.rows_, 0);
  std::fill(ret.col_sizes_, ret.col_sizes_ + ret.cols_, 0);
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

Mat Mat::operator+(const Mat &o) const {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix addition");
  Mat ret = this->addsub_alloc(o);
  for (size_t row = 0; row < ret.rows_; ++row) {
    for (size_t col_idx = 0; col_idx < ret.row_sizes_[row]; ++col_idx) {
      size_t col = ret.col(row, col_idx);
      cudla_assert_msg(col != NOT_PRESENT,
                       "tried to access column in matrix addition");
      ret(row, col) = this->operator[](row, col) + o[row, col];
    }
  }
  return ret;
}

Mat Mat::operator-(const Mat &o) const {
  cudla_assert_msg(cols_ == o.cols_ && rows_ == o.rows_,
                   "Size mismatch in matrix subtraction");
  Mat ret = this->addsub_alloc(o);
  for (size_t row = 0; row < ret.rows_; ++row) {
    for (size_t col_idx = 0; col_idx < ret.row_sizes_[row]; ++col_idx) {
      const std::optional<size_t> maybe_col = ret.col(row, col_idx);
      cudla_assert_msg(maybe_col.has_value(),
                       "tried to access column in matrix addition");
      const size_t col = maybe_col.value();
      ret(row, col) = this->operator[](row, col) - o[row, col];
    }
  }
  return ret;
}

void Mat::operator*=(float r) {
  for (size_t i = 0; i < this->n_vals_; ++i) {
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
Mat::Mat(size_t rows, size_t cols, const std::vector<MPos> &pos,
         const std::vector<float> &vals)
    : rows_(rows), cols_(cols), n_vals_(vals.size()),
      vals_(new float[vals.size()]), col_pos_(new size_t[vals.size()]),
      col_sizes_(new size_t[cols]), col_starts_(new size_t[cols]),
      row_starts_(new size_t[rows]), row_sizes_(new size_t[rows]) {
  {
    struct PosVal {
      MPos pos;
      float val;
    };

    std::fill(col_sizes_, col_sizes_ + cols_, 0);
    std::fill(row_sizes_, row_sizes_ + rows_, 0);

    cudla_assert(rows * cols >= vals.size());
    cudla_assert(pos.size() == vals.size());

    std::vector<PosVal> pvs;
    for (size_t i = 0; i < vals.size(); ++i) {
      cudla_assert(pos[i].row <= rows);
      cudla_assert(pos[i].col <= cols);

      ++row_sizes_[pos[i].row];
      ++col_sizes_[pos[i].col];
      pvs.push_back({pos[i], vals[i]});
    }

    this->init_start_arrs();
    std::sort(pvs.begin(), pvs.end(), [](const PosVal &l, const PosVal &r) {
      MPos lhs = l.pos;
      MPos rhs = r.pos;

      if (lhs.row == rhs.row) {
        return lhs.col < rhs.col;
      }

      return lhs.row < rhs.row;
    });

    for (size_t i = 0; i < vals.size(); ++i) {
      col_pos_[i] = pvs[i].pos.col;
      vals_[i] = pvs[i].val;
    }
  }
}

Mat::Mat(size_t nvals, std::vector<size_t> row_sizes,
         std::vector<size_t> col_sizes, std::vector<size_t> row_starts,
         std::vector<size_t> col_starts, std::vector<size_t> col_pos)
    : rows_(row_sizes.size()), cols_(col_sizes.size()), n_vals_(nvals),
      vals_(new float[nvals]), col_pos_(new size_t[col_pos.size()]),
      col_sizes_(new size_t[col_sizes.size()]),
      col_starts_(new size_t[col_starts.size()]),
      row_starts_(new size_t[row_starts.size()]),
      row_sizes_(new size_t[row_starts.size()]) {
  std::copy(col_starts.begin(), col_starts.end(), col_starts_);
  std::copy(col_pos.begin(), col_pos.end(), col_pos_);
  std::copy(row_starts.begin(), row_starts.end(), row_starts_);
  std::copy(row_sizes.begin(), row_sizes.end(), row_sizes_);
  std::copy(col_sizes.begin(), col_sizes.end(), col_sizes_);
}

Mat::~Mat() {
  delete[] vals_;
  delete[] col_pos_;
  delete[] col_sizes_;
  delete[] col_starts_;
  delete[] row_starts_;
  delete[] row_sizes_;
}

Mat Mat::prod_alloc(const Mat &other) const {
  cudla_assert_msg(this->cols_ == other.rows_,
                   "Size mismatch, needs this->cols_ == other.rows_.");

  size_t tmp_buf_init_cap = (this->n_vals_ + other.n_vals_) * 2;

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
      // we get a column index in a and iterate over the corresponding column
      // in B then, we set all increment all appropriate sizes this will cause
      // more values per row than we actually have
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

Mat Mat::operator*(const Mat &o) const {
  cudla_assert_msg(this->cols_ == o.rows_,
                   "Size mismatch in matrix multiplication");
  Mat ret = this->prod_alloc(o);
  std::fill(ret.vals_, ret.vals_ + n_vals_, 0.0f);

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

cudla::dense::Vec Mat::operator*(const cudla::dense::Vec &o) const {
  using cudla::dense::Vec;
  cudla_assert_msg(o.rows == this->cols_,
                   "Size mismatch in Matrix-Vector Multiplication. Cannot "
                   "mutliply Matrix of size (",
                   this->rows_, "x", this->cols_, ") and Vector of size (1x",
                   o.rows, ")");

  Vec ret(o.rows, 0.0f);
  for (size_t row = 0; row < ret.rows; ++row) {
    for (size_t col = 0; col < this->cols_; ++col) {
      ret(row) += (*this)[row, col] * o[col];
    }
  }

  return ret;
}

Mat Mat::operator*(float s) const {
  Mat ret = this->clone();
  for (size_t i = 0; i < ret.n_vals_; ++i) {
    ret.vals_[i] *= s;
  }
  return ret;
}

std::ostream &operator<<(std::ostream &stream, const Mat &mat) {
  mat.print(stream);
  return stream;
}
} // namespace cudla::sparse
