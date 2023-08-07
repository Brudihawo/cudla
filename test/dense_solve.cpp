#include "dense.h"
#include "test.h"

TEST_CASE(gauss_elim) {
  auto base = random_mat(ROWS, COLS);
  auto A = base * base.transposed();
  auto A2 = A.clone();

  const auto t = random_mat(ROWS, 1);
  const auto res = A2.gauss_elim_mut(t);

  auto actual = A * res;

  std::stringstream s;
  s << "expected: \n" << t << "\nactual: \n" << actual;
  TEST_ASSERT_MSG((almost_equal(actual, t, ERRTOL)), s.str());

  return TEST_SUCCESS;
}

TEST_CASE(gauss_elim_with_perm) {
  auto base = random_mat(ROWS, COLS);
  auto A = base * base.transposed();
  for (size_t i = 0; i < A.rows(); ++i) {
    A(i, i) = 0;
  }
  auto A2 = A.clone();

  const auto t = random_mat(ROWS, 1);
  out << "Before\n";
  out << "A\n";
  out << A << "\n";
  out << "A2\n";
  out << A2;
  const auto res = A2.gauss_elim_mut(t);

  out << "After\n";
  out << "A\n";
  out << A << "\n";
  out << "A2\n";
  out << A2;

  auto actual = A * res;

  out << "t:\n" << t << "\n";
  out << "res:\n" << res << "\n";
  out << "actual:\n" << actual << "\n";

  std::stringstream s;
  s << "expected: \n" << t << "\nactual: \n" << actual;
  // TODO: this should be wrong because of the permutations. why is it
  // correct????
  TEST_ASSERT_MSG((almost_equal(actual, t, ERRTOL)), s.str());

  return TEST_SUCCESS;
}

TEST_CASE(back_sub) {
  static constexpr size_t dim = 3;
  cudla::dense::Mat A(dim, dim, 0.0);
  for (size_t i = 0; i < dim; ++i) {
    A(i, i) = static_cast<float>(i + 1);
  }

  for (size_t i = 1; i < dim; ++i) {
    A(i - 1, i) = static_cast<float>(i + 1);
  }
  cudla::dense::Mat b(dim, 1, 1.0);
  auto x = A.back_sub(b);

  out << "A * x = b";
  out << "A:\n" << A;
  out << "b:\n" << b;
  out << "x:\n" << x;

  TEST_ASSERT(almost_equal(A * x, b, ERRTOL));

  return TEST_SUCCESS;
}

TEST_CASE(lu_solv_normal) {
  auto base = random_mat(ROWS, COLS);
  auto A = base * base.transposed();

  auto A2 = A.clone();
  const auto t = random_mat(ROWS, 1);

  auto res = A.cholesky_decomp_solv(t);
  auto actual = A2 * res;
  out << "actual:\n" << actual;
  out << "expected:\n" << t;

  TEST_ASSERT(almost_equal(t, actual, ERRTOL));

  return TEST_SUCCESS;
}
