#include <iostream>
#include <sstream>
#include <vector>

using test_fun = bool (*)(std::stringstream &cout, std::stringstream &cerr);

struct Test {
  const char *name;
  test_fun fun;
};

#define ADD_FUNC(func)                                                         \
  static Test ptr_##func __attribute((used, section("test_functions"))) = {    \
      .name = #func, .fun = func}

#define section_foreach_entry(section_name, type_t, elem)                      \
  for (type_t *elem = ({                                                       \
         extern type_t __start_##section_name;                                 \
         &__start_##section_name;                                              \
       });                                                                     \
       elem != ({                                                              \
         extern type_t __stop_##section_name;                                  \
         &__stop_##section_name;                                               \
       });                                                                     \
       ++elem)

#define TEST_CASE(name)                                                        \
  bool(name)(auto out, auto err);                                              \
  ADD_FUNC(name);                                                              \
  bool(name)([[maybe_unused]] auto out, [[maybe_unused]] auto err)

#define TEST_ASSERT(cond) TEST_ASSERT_(cond, __FILE__, __LINE__)
#define TEST_ASSERT_(cond, file, line)                                         \
  do {                                                                         \
    if (!(cond)) {                                                             \
      err << file << ":" << line << ": TEST_ASSERT(" #cond ") failed";         \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct TestResult {
  const char *name;
  bool success_;
  std::string out_;
  std::string err_;

  TestResult(const Test &test) : name(test.name) {
    std::stringstream out;
    std::stringstream err;
    success_ = test.fun(out, err);
    out_ = out.str();
    err_ = err.str();
  }
};
