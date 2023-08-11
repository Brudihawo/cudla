#pragma once
#include "errors.h"
#include "permutations.h"
#include <iomanip>
#include <optional>
#include <vector>

namespace cudla::sparse {
struct MPos {
  size_t row, col;
};

class Mat {
public:
  /** @brief initialize empty matrix
   */
  Mat(size_t rows, size_t cols, size_t n_vals);

  Mat(size_t rows, size_t cols, const std::vector<MPos> &pos,
      const std::vector<float> &vals);

  /** @brief initialize empty matrix with given non-zero structure
   */
  Mat(size_t nvals, std::vector<size_t> row_sizes,
      std::vector<size_t> col_sizes, std::vector<size_t> row_starts,
      std::vector<size_t> col_starts, std::vector<size_t> col_pos);

  ~Mat();

  // TODO: implement rest of the constructors from C version of la

  /** @brief clone matrix structure but not it's values
   */
  Mat clone_empty() const;

  /** @brief clone matrix structure and values
   */
  Mat clone() const;

  size_t rows() const;
  size_t cols() const;

  size_t idx(size_t row, size_t col) const;
  bool has_loc(size_t row, size_t col) const;

  /**
   * @brief return transposed version of self
   *
   * @return self transposed version of self
   */
  Mat transposed() const;

  // TODO: implement transpose_in_place

  // Accessors
  /** @brief Get copy of value at location
   */
  float operator[](size_t row, size_t col) const;

  /** @brief Get reference to value at location
   */
  float &operator()(size_t row, size_t col);

  /** @brief elementwise equality with other matrix
   */
  bool operator==(const Mat &o) const;

  /** @brief adds other matrix
   *  @note this may change the non-zero structure and thus needs to allocate
   *  a new matrix
   */
  Mat operator+(const Mat &o) const;

  /** @brief subtracts other matrix
   *  @note this may change the non-zero structure and thus needs to allocate
   *  a new matrix
   */
  Mat operator-(const Mat &o) const;

  /** @brief perform a matrix multiplication with o
   *  @note this computes size requirements and allocates a return matrix.
   *        For many matrix mutliplications requiring the same size, size
   *        requirements should be calculated once and then repeatedly assigned
   *        into a preallocated matrix
   *
   *        TODO: implement multiplication into preallocated
   */
  Mat operator*(const Mat &o) const;

  /** @brief scale matrix elementwise by scalar
   */
  Mat operator*(float s) const;

  /** @brief scale matrix by scalar inplace
   */
  void operator*=(float);

  /** @brief check if self and other have the same non-zero structure
   */
  bool structure_eq(const Mat &other) const;

  Mat cholesky_decomp_solv(const Mat &b);

  void print(auto &out) const {
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

  void print_shape(auto &out) const {
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        if (this->has_loc(row, col)) {
          out << "#";
        } else {
          out << " ";
        }
      }
      out << "\n";
    }
  }

  friend std::ostream &operator<<(std::ostream &stream, const Mat &mat);

  size_t rows_ = 0, cols_ = 0;
  size_t n_vals_;

  // TODO: manage memory by myself instead of using vector
  float *vals_;        //< length: n_vals
  size_t *col_pos_;    //< length: n_vals_
  size_t *col_sizes_;  //< length: cols_
  size_t *col_starts_; //< length: cols_
  size_t *row_starts_; //< length: rows_
  size_t *row_sizes_;  //< length: rows_

  // TODO: do i want a bounds-check here?
  inline bool row_empty(size_t row) const { return row_sizes_[row] == 0; }

  Mat addsub_alloc(const Mat &other) const;
  Mat prod_alloc(const Mat &other) const;

  void init_start_arrs();
  size_t col(size_t row, size_t col_idx) const;
  size_t col_or_panic(size_t row, size_t col_idx) const;
  static constexpr size_t NOT_PRESENT = SIZE_MAX;
};
std::optional<size_t> binary_search(const size_t *const arr, size_t size,
                                    size_t val);

} // namespace cudla::sparse
