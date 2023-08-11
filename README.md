# CudLA
Linear algebra library in c++, but with support for performing operations on the gpu

## Error handling
The library checks if matrix sizes for multiplication and addition match at runtime.
The macros `cudla_assert` and `cudla_assert_msg` can be overwritten before `#include`-ing any CudLA header to completely turn off runtime asserts. Alternatively, the macro `CUDLA_ERR_OSTREAM` can be overwritten to change the assert message destination. CudLA functions generally do not throw exceptions.


# Roadmap
- [ ] implement dense solving algorithms (port from c)
- [ ] factor out solving into solver classes
- [ ] implement sparse matrices (port from c)
- [ ] implement GPU multiplication
- [ ] implement GPU solving
