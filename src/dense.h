#pragma once
#include "errors.h"
#include "permutations.h"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>

namespace cudla::dense {

class Mat {
private:
  size_t rows_ = 0, cols_ = 0;
  std::vector<float> vals_;

  size_t idx(size_t row, size_t col) const;

  void swap_rows(size_t i, size_t j, std::vector<float> &cpybuf);

public:
  /** @brief create zero-initialized matrix with given dimensions
   */
  Mat(size_t rows, size_t cols);

  /** @brief create initialized matrix with given dimensions and values
   * initialized to val
   */
  Mat(size_t rows, size_t cols, float val);

  /**
   * @brief create a matrix and populate entries using a predicate
   *
   * @param rows number of rows in matrix
   * @param cols number of columns in matrix
   * @param pred predicate (row, column) -> value for assigning elements
   */
  Mat(size_t rows, size_t cols, std::function<float(size_t, size_t)> pred);

  Mat clone_empty() const;
  Mat clone() const;

  size_t rows() const;
  size_t cols() const;

  /**
   * @brief return transposed version of self
   *
   * @return self transposed version of self
   */
  Mat transposed() const;
  Mat make_triu() const;
  float det() const;

  // TODO: implement transpose_in_place

  // Accessors
  float operator[](size_t row, size_t col) const;
  float &operator()(size_t row, size_t col);

  bool operator==(const Mat &o);
  Mat operator+(const Mat &o);
  Mat operator-(const Mat &o);
  Mat operator*(const Mat &o);

  Mat back_sub(const Mat &b) const;

  /**
   * @brief solve the system of equations this * x = b
   * this needs to be a square matrix and target needs to be a column vector
   * this matrix is modified to a triangular upper matrix
   *
   * @param target right hand side of the system of equations
   * @return x - solution to system of equations
   */
  Mat gauss_elim_mut(const Mat &b);
  std::optional<util::RowPerms> cholesky_decomp_mut();
  void row_permute(const util::RowPerms &perms);
  Mat lu_forw_sub(const Mat &b);
  Mat cholesky_decomp_solv(const Mat &b);

  void print(std::ostream &out) const {
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        const float val = (*this)[row, col];
        out << std::setw(8) << std::setprecision(3) << val;
        if (col < cols_ - 1) {
          out << ", ";
        }
      }
      out << "\n";
    }
  }
  friend std::ostream &operator<<(std::ostream &stream, const Mat &mat);
};

} // namespace cudla::dense
