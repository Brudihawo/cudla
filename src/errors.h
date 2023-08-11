#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CUDLA_DEFAULT_ERR_OSTREAM std::cerr

#ifndef CUDLA_ERR_OSTREAM
#define CUDLA_ERR_OSTREAM CUDLA_DEFAULT_ERR_OSTREAM
#endif

#ifdef ASSERT_FILE_LOCS
#define cudla_assert_msg(expr, msg)                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      CUDLA_ERR_OSTREAM << __FILE__ << ":" << __LINE__ << ": ";                \
    }                                                                          \
    cudla_assert_msg_(expr, msg);                                              \
  } while (0)
#define cudla_assert(expr)                                                     \
  do {                                                                         \
    if (!(expr)) {                                                             \
      CUDLA_ERR_OSTREAM << __FILE__ << ":" << __LINE__ << ": ";                \
    }                                                                          \
    cudla_assert_(expr);                                                       \
  } while (0)
#else
#define cudla_assert_msg(expr, msg) cudla_assert_msg_(expr, msg)
#define cudla_assert(expr) cudla_assert_(expr)
#endif

#define cudla_assert_msg_(expr, msg)                                           \
  do {                                                                         \
    if (!(expr)) {                                                             \
      CUDLA_ERR_OSTREAM << "CudLA assertion failed: '" << #expr << ": '"       \
                        << msg << "''. Call stack: \n";                        \
      cudla::err::print_backtrace(CUDLA_ERR_OSTREAM);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define cudla_assert_(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      CUDLA_ERR_OSTREAM << "CudLA assertion failed: '" << #expr                \
                        << "'. Call stack:\n";                                 \
      cudla::err::print_backtrace(CUDLA_ERR_OSTREAM);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace cudla::err {
static constexpr uint16_t STACKTRACE_CAP = 64;
static constexpr uint16_t BUF_SIZE = 512;
void print_backtrace(auto &stream) {
  void *symbols[STACKTRACE_CAP];
  char buf[BUF_SIZE];

  int32_t size = backtrace(symbols, STACKTRACE_CAP);
  char **strings = backtrace_symbols(symbols, size);

  if (size < 1) {
    return;
  }

  for (uint16_t i = 1; i < size; ++i) {
    stream << "# " << i - 1 << ": ";
    char *begin = strings[i];
    char *file_end = std::strchr(strings[i], '(');
    std::copy(begin, file_end, std::ostream_iterator<char>(stream));

    char *mangled_fn_end = std::strchr(file_end, '+');

    stream << " ";

    int status = 0;
    std::fill(buf, buf + BUF_SIZE, 0);
    if (static_cast<size_t>(mangled_fn_end - file_end - 1) > sizeof(buf)) {
      std::copy(file_end + 1, file_end + BUF_SIZE, buf);
    } else {
      std::copy(file_end + 1, mangled_fn_end, buf);
    }

    const char *demangled = abi::__cxa_demangle(buf, nullptr, nullptr, &status);
    if (status) {
      std::copy(file_end + 1, mangled_fn_end,
                std::ostream_iterator<char>(stream));
    } else {
      stream << demangled;
      delete[] demangled;
    }

    stream << " ";

    char *offset_end = std::strchr(mangled_fn_end, ')');
    std::copy(mangled_fn_end, offset_end, std::ostream_iterator<char>(stream));

    stream << " @ ";
    char *address_begin = std::strchr(offset_end, '[');
    char *address_end = std::strchr(address_begin, ']');
    std::copy(address_begin + 1, address_end,
              std::ostream_iterator<char>(stream));
    stream << "\n";
  }

  delete[] strings;
}
} // namespace cudla::err
