
#include "field_pulse.h"

#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/maths.h>

#include <fstream>
#include <memory>
#include <jams/helpers/output.h>

namespace {


    class GaussianSurfacePulse : public TemporalFieldPulse {
    public:
        GaussianSurfacePulse(
            const double& surface_cutoff, const double& temporal_width,
            const double& temporal_center, const Vec3& field) :
            surface_cutoff_(surface_cutoff),
            temporal_width_(temporal_width),
            temporal_center_(temporal_center),
            field_(field) {};

        explicit GaussianSurfacePulse(const libconfig::Setting &settings)
        : GaussianSurfacePulse(
            jams::config_required<double>(settings, "surface_cutoff"),
            jams::config_required<double>(settings, "temporal_width"),
            jams::config_required<double>(settings, "temporal_center"),
            jams::config_required<Vec3>(settings, "field")) {};

        Vec3 local_field(const double& time, const Vec3& r) override {
          if (!count_as_surface(r)) {
            return {0.0, 0.0, 0.0};
          }
          return max_field(time);
        }

        // Returns the maximum field at a given time and internally caches the
        // result. This avoid constant recalculation of the gaussian function
        // for function calls all at the same timestep.
        Vec3 max_field(const double& time) override {
          if (time == cache_time_) {
            return cache_field_;
          }
          cache_time_ = time;
          cache_field_ = gaussian(time, temporal_center_, 1.0, temporal_width_) * field_;
          return cache_field_;
        }

        // Returns true if the z-component of position r is greater than the
        // surface cutoff plane.
        bool count_as_surface(const Vec3& r) const {
          return r[2] > surface_cutoff_;
        }

    private:
        double surface_cutoff_;
        double temporal_center_;
        double temporal_width_;
        Vec3 field_;
        Vec3 cache_field_ = {0.0, 0.0, 0.0};
        double cache_time_ = std::numeric_limits<double>::quiet_NaN();
    };


    class SincPulse : public TemporalFieldPulse {
    public:
        SincPulse(
            const double& temporal_width,
            const double& temporal_center, const Vec3& field) :
            temporal_center_(temporal_center),
            temporal_width_(temporal_width),
            field_(field) {};

        explicit SincPulse(const libconfig::Setting &settings)
            : SincPulse(
            jams::config_required<double>(settings, "temporal_width"),
            jams::config_required<double>(settings, "temporal_center"),
            jams::config_required<Vec3>(settings, "field")) {};

        Vec3 local_field(const double& time, const Vec3& r) override {
          return max_field(time);
        }

        // Returns the maximum field at a given time and internally caches the
        // result. This avoid constant recalculation of the gaussian function
        // for function calls all at the same timestep.
        Vec3 max_field(const double& time) override {
          if (time == cache_time_) {
            return cache_field_;
          }

          cache_time_ = time;

          double x = temporal_width_ * (time - temporal_center_);
          cache_field_ = (sin(x) / x) * field_;

          return cache_field_;
        }

    private:
        double temporal_center_;
        double temporal_width_;
        Vec3 field_;
        Vec3 cache_field_ = {0.0, 0.0, 0.0};
        double cache_time_ = std::numeric_limits<double>::quiet_NaN();
    };
}

FieldPulseHamiltonian::FieldPulseHamiltonian(const libconfig::Setting &settings,
                                             unsigned int size) : Hamiltonian(
    settings, size) {


  auto pulse_type = lowercase(jams::config_required<std::string>(settings, "pulse_type"));

  if (pulse_type == "gaussian_surface_pulse") {
    temporal_field_pulse_ = std::make_unique<GaussianSurfacePulse>(settings);
  }
  else if (pulse_type == "sinc_pulse") {
    temporal_field_pulse_ = std::make_unique<SincPulse>(settings);
  } else {
    throw std::runtime_error("Unknown field pulse type " + pulse_type);
  }

  std::ofstream pulse_file(jams::output::full_path_filename("field_pulse.tsv"));
  output_pulse(pulse_file);
}

double FieldPulseHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  calculate_energies(time);
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void FieldPulseHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
      energy_(i) = calculate_energy(i, time);
  }
}

void FieldPulseHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field(i, time);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

Vec3 FieldPulseHamiltonian::calculate_field(int i, double time) {
  Vec3 r = ::globals::lattice->atom_position(i);
  return globals::mus(i) *
      temporal_field_pulse_->local_field(time, r);
}

double FieldPulseHamiltonian::calculate_energy(int i, double time) {
  Vec3 spin = {globals::s(i,0), globals::s(i, 1), globals::s(i, 2)};
  Vec3 field = calculate_field(i, time);
  return -dot(spin, field);
}

double FieldPulseHamiltonian::calculate_energy_difference(int i,
                                                          const Vec3 &spin_initial,
                                                          const Vec3 &spin_final, double time) {
  const auto e_initial = -dot(spin_initial, calculate_field(i, time));
  const auto e_final = -dot(spin_final, calculate_field(i, time));

  return (e_final - e_initial);
}

void FieldPulseHamiltonian::output_pulse(std::ofstream& os) {
  if (!os.is_open()) {
    os.open(jams::output::full_path_filename("field_pulse.tsv"));
    os << "time  Hx  Hy  Hz\n";
  }


  for (auto i = 0; i < globals::solver->max_steps(); ++i) {
    auto time = i * globals::solver->time_step();
    auto field = temporal_field_pulse_->max_field(time);
    os << jams::fmt::sci << time;
    os << jams::fmt::decimal << field[0];
    os << jams::fmt::decimal << field[1];
    os << jams::fmt::decimal << field[2] << "\n";
  }
}
