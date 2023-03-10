#include "test.h"
#include "dense.h"
#include <random>

#define ROWS 15
#define COLS 15
#define ERRTOL 0.0001

#define RED "\033[31;1m"
#define GRN "\033[32;1m"
#define RST "\033[0;0m"

cudla::dense::Mat<double> random_mat(size_t rows, size_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  return {rows, cols, [&dis, &gen](size_t, size_t) { return dis(gen); }};
}

template <typename T>
bool almost_equal(const cudla::dense::Mat<T> &a, const cudla::dense::Mat<T> &b,
                  T errtol) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }

  for (size_t row = 0; row < b.rows(); ++row) {
    for (size_t col = 0; col < b.cols(); ++col) {
      if (std::abs(a[row, col] - b[row, col]) > errtol) {
        return false;
      }
    }
  }
  return true;
}

TEST_CASE(clone) {
  cudla::dense::Mat<double> mat(ROWS, COLS);
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<double>(col + row * COLS);
    }
  }
  auto mat2 = mat.clone();

  TEST_ASSERT(mat2.operator==(mat));

  return TEST_SUCCESS;
}

TEST_CASE(print) {
  cudla::dense::Mat<double> mat(ROWS, COLS);
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<double>(col + row * COLS);
    }
  }
  mat.print(out);
  return TEST_SUCCESS;
}

TEST_CASE(inv_mul_basic) {
  cudla::dense::Mat<double> mat(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 2;
  }

  cudla::dense::Mat<double> inv(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 0.5;
  }

  cudla::dense::Mat<double> expected(ROWS, COLS, 0);
  for (size_t i = 0; i < ROWS; ++i) {
    mat(i, i) = 1;
  }

  auto ret = mat * inv;
  std::stringstream s;
  s << "mat: \n" << mat << "\ninv: \n" << inv << "\nret" << ret;
  TEST_ASSERT_MSG(ret.operator==(expected), s.str());

  return TEST_SUCCESS;
}

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

int main() {
  std::vector<TestResult> results;
  std::vector<Test> tests;

  section_foreach_entry(test_functions, Test, entry) {
    tests.push_back(*entry);
  }

  for (auto &test : tests) {
    results.emplace_back(test);
  }

  for (const auto &entry : results) {
    if (!entry.success_) {
      std::cout << RED " [ F ] " RST << entry.name << "\n";
      if (!entry.out_.empty()) {
        std::cout << "======== STDOUT ========\n" << entry.out_ << "\n";
        std::cout << "========================\n";
      }

      if (!entry.err_.empty()) {
        std::cout << "======== STDERR ========\n" << entry.err_ << "\n";
        std::cout << "========================\n";
      }
    } else {
      std::cout << GRN " [ P ] " RST << entry.name << "\n";
    }
  }
}
