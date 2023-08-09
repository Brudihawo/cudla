#include "errors.h"
#include "permutations.h"
#include <optional>
#include <vector>

namespace cudla::sparse {

class Mat;

// template <typename T, size_t rows, size_t cols>
// concept CompatibleMatrix = requires () {}

class Mat {
public:
  Mat(size_t rows, size_t cols, size_t n_vals);

  Mat(size_t nvals, std::vector<size_t> row_sizes,
      std::vector<size_t> col_sizes, std::vector<size_t> row_starts,
      std::vector<size_t> col_starts, std::vector<size_t> col_pos);
  Mat clone_empty() const;

  Mat clone() const;

  size_t rows() const;
  size_t cols() const;

  std::optional<size_t> idx(size_t row, size_t col) const;
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
   *
   */
  float operator[](size_t row, size_t col) const;

  /** @brief Get reference to value at location
   *
   */
  float &operator()(size_t row, size_t col);

  bool operator==(const Mat &o);
  Mat operator+(const Mat &o);
  Mat operator-(const Mat &o);
  Mat operator*(const Mat &o);

  void operator*=(float);

  bool structure_eq(const Mat &other) const;

  Mat make_triu() const;

  float det() const;

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

  void print(auto &out) const;

  friend std::ostream &operator<<(std::ostream &stream, const Mat &mat);

private:
  struct Pos {
    size_t row, col;
  };

  size_t rows_ = 0, cols_ = 0;
  std::vector<float> vals_;

  std::vector<size_t> col_sizes_;
  std::vector<size_t> col_starts_;
  std::vector<size_t> col_pos_;
  std::vector<size_t> row_starts_;
  std::vector<size_t> row_sizes_;

  // TODO: do i want a bounds-check here?
  inline bool row_empty(size_t row) const { return row_sizes_[row] == 0; }

  Mat addsub_alloc(const Mat &other) const;
  Mat prod_alloc(const Mat &other) const;

  void init_start_arrs();
  std::optional<size_t> col(size_t row, size_t col_idx) const;
};

} // namespace cudla::sparse
