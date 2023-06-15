#include <jams/physics/induced_spin_pulse.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/cuda/cuda_spin_ops.h>

InducedSpinPulsePhysics::InducedSpinPulsePhysics(const libconfig::Setting &settings)
: Physics(settings)
, pulse_center_time_(jams::config_required<double>(settings, "pulse_center_time"))
, pulse_width_(jams::config_required<double>(settings, "pulse_width"))
, pulse_height_(jams::config_required<double>(settings, "pulse_height")) {

  std::string material = jams::config_required<std::string>(settings, "material");

  std::vector<int> indices;
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (globals::lattice->atom_material_name(i) == material) {
      indices.push_back(i);
    }
  }

  spin_indices_ = jams::MultiArray<int, 1>(indices.begin(), indices.end());

}


void InducedSpinPulsePhysics::update(const int &iterations, const double &time,
                                     const double &dt) {
  const double relative_time = (time - pulse_center_time_);

  double moment = 0.0;
  if (std::abs(relative_time) <= 10*pulse_width_) {
    moment = pulse_height_*(1.0/(std::sqrt(kTwoPi)*pulse_width_))*exp(-0.5*square(relative_time/pulse_width_));
  }

  jams::add_to_spins_cuda(globals::s, moment, spin_indices_);
}