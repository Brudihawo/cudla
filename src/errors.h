#pragma once

#include <cstdint>
#include <cstdlib>
#include <execinfo.h>
#include <stdlib.h>

#define CUDLA_DEFAULT_ERR_OSTREAM std::cerr

#define cudla_assert_msg(expr, msg)                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "CudLA assertion failed: '" << #expr << ": " << msg         \
                << "'. Call stack:\n";                                         \
      cudla::err::print_backtrace(CUDLA_DEFAULT_ERR_OSTREAM);                  \
    }                                                                          \
  } while (0)

#define cudla_assert(expr)                                                     \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "CudLA assertion failed: '" << #expr << "'. Call stack:\n"; \
      cudla::err::print_backtrace(CUDLA_DEFAULT_ERR_OSTREAM);                  \
    }                                                                          \
  } while (0)

namespace cudla::err {
static constexpr uint16_t STACKTRACE_CAP = 64;
void print_backtrace(auto &ostream) {
  void *symbols[STACKTRACE_CAP];

  int32_t size = backtrace(symbols, STACKTRACE_CAP);
  char **strings = backtrace_symbols(symbols, size);

  if (size < 1) {
    return;
  }

  for (uint16_t i = 1; i < size; ++i) {
    ostream << "#" << i-1 << ": " << strings[i] << "\n";
  }

  delete[] strings;
}

} // namespace cudla::err
