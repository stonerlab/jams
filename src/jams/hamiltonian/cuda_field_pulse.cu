#include <jams/hamiltonian/cuda_field_pulse.h>
#include <jams/core/globals.h>
#include <jams/helpers/output.h>
#include <jams/core/solver.h>
#include <jams/core/lattice.h>
#include <jams/helpers/maths.h>

__global__ void cuda_field_pulse_surface_kernel(const unsigned int num_spins, const double surface_cutoff, const jams::Real * dev_mus, const jams::Real * dev_r, const jams::Real3 b_field, jams::Real * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = 3u * idx;
  if (idx >= num_spins) return;

  // check z component of spin position
  if (dev_r[base + 2] > surface_cutoff) {
    dev_h[base + 0] = dev_mus[idx] * b_field.x;
    dev_h[base + 1] = dev_mus[idx] * b_field.y;
    dev_h[base + 2] = dev_mus[idx] * b_field.z;
  } else {
    dev_h[base + 0] = 0.0;
    dev_h[base + 1] = 0.0;
    dev_h[base + 2] = 0.0;
  }
}

CudaFieldPulseHamiltonian::CudaFieldPulseHamiltonian(
    const libconfig::Setting &settings, unsigned int size) : Hamiltonian(
    settings, size) {

  surface_cutoff_ = jams::config_required<jams::Real>(settings, "surface_cutoff");
  temporal_width_ = jams::config_required<jams::Real>(settings, "temporal_width");
  temporal_center_ = jams::config_required<jams::Real>(settings, "temporal_center");
  max_field_ = jams::config_required<jams::Vec<jams::Real, 3>>(settings, "field");

  positions_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      positions_(i, j) = globals::lattice->lattice_site_position_cart(i)[j];
    }
  }

  output_pulse();
}

void CudaFieldPulseHamiltonian::calculate_fields(jams::Real time) {

  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  jams::Vec<jams::Real, 3> b_field = gaussian(time, temporal_center_, static_cast<jams::Real>(1), temporal_width_) * max_field_;
  jams::Real3 field = {b_field[0], b_field[1], b_field[2]};
  cuda_field_pulse_surface_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
      (globals::num_spins, surface_cutoff_, globals::mus.device_data(), positions_.device_data(),
       field, field_.mutable_device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaFieldPulseHamiltonian::output_pulse() const {
  jams::output::TsvWriter tsv(
      jams::output::hamiltonian_filename(name(), "tsv"),
      {{"time", "picoseconds"},
       {"Hx", "T", jams::output::ColFmt::Fixed},
       {"Hy", "T", jams::output::ColFmt::Fixed},
       {"Hz", "T", jams::output::ColFmt::Fixed}});

  for (auto i = 0; i < globals::solver->max_steps(); ++i) {
    jams::Real time = i * globals::solver->time_step();
    jams::Vec<jams::Real, 3> field = gaussian(time, temporal_center_, static_cast<jams::Real>(1), temporal_width_) * max_field_;
    tsv.write_row_values(time, field[0], field[1], field[2]);
  }
}
