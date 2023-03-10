#pragma once
#include "errors.h"
#include <cstdlib>
#include <iostream>
#include <vector>

namespace cudla::util {
struct RowPerms {
  std::vector<size_t> order_;
  size_t n_swaps_ = 0;
  size_t rows_ = 0;

public:
  RowPerms(size_t rows) : order_(rows), rows_(rows) {
    for (size_t i = 0; i < rows_; ++i) {
      order_[i] = i;
    }
  }

  void add_perm(size_t i, size_t j) {
    cudla_assert_msg(i < rows_, "row index out of bounds");
    cudla_assert_msg(j < rows_, "row index out of bounds");
    size_t tmp = order_[j];
    order_[j] = order_[i];
    order_[i] = tmp;
    n_swaps_++;
  }
};
} // namespace cudla::util
