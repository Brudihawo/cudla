#include "test.h"
#include "mat.h"

#define ROWS 3
#define COLS 3

#define RED "\033[31;1m"
#define GRN "\033[32;1m"
#define RST "\033[0;0m"

TEST_CASE(clone) {
  dense::Mat<double, ROWS, COLS> mat;
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<double>(col + row * COLS);
    }
  }
  auto mat2 = mat.clone();

  TEST_ASSERT(mat2.operator==(mat));

  return true;
}

TEST_CASE(print) {
  dense::Mat<double, ROWS, COLS> mat;
  for (size_t row = 0; row < ROWS; ++row) {
    for (size_t col = 0; col < ROWS; ++col) {
      mat(row, col) = static_cast<double>(col + row * COLS);
    }
  }
  mat.print(out);
  return true;
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
      std::cout << GRN " [ S ] " RST << entry.name << "\n";
    }
  }
}
