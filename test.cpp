#include "test.h"
#include "dense.h"
#include "test_dense_solve.c"

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
    std::stringstream descs;
    descs << entry.file << ":" << entry.line << ": " << entry.name;
    std::string desc = descs.str();
    if (!entry.success_) {
      std::cout << RED " [ F ] " RST << desc << "\n";
      if (!entry.out_.empty()) {
        std::cout << "======== STDOUT ========\n" << entry.out_ << "\n";
        std::cout << "========================\n";
      }

      if (!entry.err_.empty()) {
        std::cout << "======== STDERR ========\n" << entry.err_ << "\n";
        std::cout << "========================\n";
      }
    } else {
      std::cout << GRN " [ P ] " RST << desc << "\n";
    }
  }
}
