#pragma once
#include <iostream>
#include <memory>

namespace dense {

template <typename T, size_t rows, size_t cols> class Mat {
private:
  T *vals_;

  size_t idx(size_t row, size_t col) const { return row * cols + col; }

public:
  Mat() : vals_(new T[rows * cols]) {}
  ~Mat() { delete[] vals_; }

  Mat(T val) : vals_(new T[rows * cols]) {
    for (auto &i : vals_) {
      i = val;
    }
  }

  Mat<T, rows, cols> clone_empty() const { return Mat<T, rows, cols>(); }
  Mat<T, rows, cols> clone() const {
    Mat<T, rows, cols> ret;
    for (size_t i = 0; i < rows * cols; ++i) {
      ret.vals_[i] = vals_[i];
    }
    return ret;
  }

  Mat<T, rows, cols> transpose() const {
    Mat<T, rows, cols> ret;
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        ret.vals_[ret.idx(row, col)] = vals_[idx(col, row)];
      }
    }
  }

  T operator[](size_t row, size_t col) const { return vals_[idx(row, col)]; }
  T &operator()(size_t row, size_t col) { return vals_[idx(row, col)]; }

  template <size_t o_rows, size_t o_cols>
  bool operator==(const Mat<T, o_rows, o_cols> &other) {
    if (o_rows != rows || o_cols != cols) {
      return false;
    }

    for (size_t i = 0; i < rows * cols; ++i) {
      if (other.vals_[i] != vals_[i]) {
        return false;
      }
    }
    return true;
  }

  Mat<T, rows, cols> make_triu() const {
    static_assert(rows == cols, "matrix needs to be square");
    auto ret = this->clone();
    ret.print();

    for (size_t zero_col = 0; zero_col < cols - 1;
         zero_col++) { // zeroing of i-th column
      for (size_t row = zero_col + 1; row < rows; row++) {   // row n
        for (size_t col = zero_col + 1; col < cols; col++) { // column j
          ret(row, col) -=
              ret[row, zero_col] / ret[zero_col, zero_col] * ret[zero_col, col];
        }
      }
    }

    for (size_t i = 1; i < rows; i++) {
      for (size_t j = 0; j < i; j++) {
        ret(i, j) = 0.0f;
      }
    }
    return ret;
  }

  T det() const {
    auto tmp = this->make_triu();
    T ret = 0.0;
    for (size_t i = 0; i < rows; ++i) {
      ret *= tmp[i, i];
    }
    return ret;
  }

  void print(auto &out) const {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const T val = (*this)[row, col];
        out << val;
        if (col < cols - 1) {
          out << ",";
        }
      }
      out << "\n";
    }
  }
};

} // namespace dense
