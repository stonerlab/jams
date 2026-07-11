// spin_output.h                                                    -*-C++-*-

#ifndef JAMS_MONITORS_SPIN_OUTPUT_H
#define JAMS_MONITORS_SPIN_OUTPUT_H

#include <jams/containers/multiarray.h>
#include <jams/containers/vec.h>
#include <jams/core/globals.h>

namespace jams::monitors {

template <typename SpinView, typename MomentView>
[[nodiscard]] inline jams::Vec<double, 3> output_spin_vector(
    const SpinView& spins,
    const MomentView& moments,
    const int spin) {
  if (static_cast<double>(moments(spin)) == 0.0) {
    return {0.0, 0.0, 0.0};
  }

  return {spins(spin, 0), spins(spin, 1), spins(spin, 2)};
}

template <typename SourceSpinIndex>
[[nodiscard]] inline jams::MultiArray<double, 2> make_spin_output_array(
    const int point_count,
    SourceSpinIndex source_spin_index) {
  jams::MultiArray<double, 2> output(point_count, 3);
  const auto spins = globals::s.host_view();
  const auto moments = globals::mus.host_view();
  auto output_view = output.mutable_host_view();

  for (auto output_index = 0; output_index < point_count; ++output_index) {
    const auto spin = source_spin_index(output_index);
    const auto spin_output = output_spin_vector(spins, moments, spin);
    for (auto component = 0; component < 3; ++component) {
      output_view(output_index, component) = spin_output[component];
    }
  }

  return output;
}

[[nodiscard]] inline jams::MultiArray<double, 2> make_spin_output_array() {
  return make_spin_output_array(globals::num_spins, [](const int spin) {
    return spin;
  });
}

}  // namespace jams::monitors

#endif  // JAMS_MONITORS_SPIN_OUTPUT_H
