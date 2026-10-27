#ifndef JAMS_HELPERS_FINITE_H
#define JAMS_HELPERS_FINITE_H

#include <bit>
#include <cstdint>

#ifndef JAMS_HOST_DEVICE
#if defined(__CUDACC__)
#define JAMS_HOST_DEVICE __host__ __device__
#else
#define JAMS_HOST_DEVICE
#endif
#endif

namespace jams {

// Inspect the exponent so validation remains effective in fast-math builds.
JAMS_HOST_DEVICE inline bool is_finite(const double value) {
#if defined(__CUDA_ARCH__)
  const auto bits = static_cast<std::uint64_t>(__double_as_longlong(value));
#else
  // A volatile integer load prevents LLVM folding this back into an FP
  // finiteness test, which -ffast-math would then assume is always true.
  const volatile auto bits = std::bit_cast<std::uint64_t>(value);
#endif
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

}  // namespace jams
#endif
