#include <jams/hamiltonian/cuda_field_pulse.h>
#include <jams/hamiltonian/cuda_field_pulse_kernel.cuh>

#include <jams/core/globals.h>
#include <jams/helpers/output.h>
#include <jams/core/solver.h>
#include <jams/core/lattice.h>
#include <jams/helpers/maths.h>
#include <fstream>

CudaFieldPulseHamiltonian::CudaFieldPulseHamiltonian(
    const libconfig::Setting &settings, unsigned int size) : Hamiltonian(
    settings, size) {

  surface_cutoff_ = jams::config_required<double>(settings, "surface_cutoff");
  temporal_width_ = jams::config_required<double>(settings, "temporal_width");
  temporal_center_ = jams::config_required<double>(settings, "temporal_center");
  max_field_ = jams::config_required<Vec3>(settings, "field");

  positions_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      positions_(i, j) = lattice->atom_position(i)[j];
    }
  }

  std::ofstream pulse_file(jams::output::full_path_filename("field_pulse.tsv"));
  output_pulse(pulse_file);
}

void CudaFieldPulseHamiltonian::calculate_fields(double time) {

  dim3 block_size;
  block_size.x = 64;

  dim3 grid_size;
  grid_size.x = (globals::num_spins + block_size.x - 1) / block_size.x;

  Vec3 b_field = gaussian(time, temporal_center_, 1.0, temporal_width_) * max_field_;
  double3 field = {b_field[0], b_field[1], b_field[2]};
  cuda_field_pulse_surface_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>>
      (globals::num_spins, surface_cutoff_, globals::mus.device_data(), positions_.device_data(),
       field, field_.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaFieldPulseHamiltonian::output_pulse(std::ofstream& os) {
  if (!os.is_open()) {
    os.open(jams::output::full_path_filename("field_pulse.tsv"));
    os << "time  Hx  Hy  Hz\n";
  }


  for (auto i = 0; i < solver->max_steps(); ++i) {
    auto time = i * solver->time_step();
    auto field = gaussian(time, temporal_center_, 1.0, temporal_width_) * max_field_;
    os << jams::fmt::sci << time;
    os << jams::fmt::decimal << field[0];
    os << jams::fmt::decimal << field[1];
    os << jams::fmt::decimal << field[2] << "\n";
  }
}
