#include "test.h"
#include "dense.h"
#include "dense_solve.cpp"
#include "dense_basic.cpp"
#include "sparse_basic.cpp"

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
