
#include "field_pulse.h"

#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/maths.h>

#include <fstream>
#include <memory>
#include <jams/helpers/output.h>
#include <jams/core/globals.h>

namespace {


    class GaussianSurfacePulse : public TemporalFieldPulse {
    public:
        GaussianSurfacePulse(
            const jams::Real& surface_cutoff, const jams::Real& temporal_width,
            const jams::Real& temporal_center, const Vec3R& field) :
            surface_cutoff_(surface_cutoff),
            temporal_width_(temporal_width),
            temporal_center_(temporal_center),
            field_(field) {};

        explicit GaussianSurfacePulse(const libconfig::Setting &settings)
        : GaussianSurfacePulse(
            jams::config_required<jams::Real>(settings, "surface_cutoff"),
            jams::config_required<jams::Real>(settings, "temporal_width"),
            jams::config_required<jams::Real>(settings, "temporal_center"),
            jams::config_required<Vec3R>(settings, "field")) {};

        Vec3R local_field(const jams::Real& time, const Vec3R& r) override {
          if (!count_as_surface(r)) {
            return {0.0, 0.0, 0.0};
          }
          return max_field(time);
        }

        // Returns the maximum field at a given time and internally caches the
        // result. This avoid constant recalculation of the gaussian function
        // for function calls all at the same timestep.
        Vec3R max_field(const jams::Real& time) override {
          if (time == cache_time_) {
            return cache_field_;
          }
          cache_time_ = time;
          cache_field_ = gaussian(time, temporal_center_, jams::Real(1.0), temporal_width_) * field_;
          return cache_field_;
        }

        // Returns true if the z-component of position r is greater than the
        // surface cutoff plane.
        bool count_as_surface(const Vec3R& r) const {
          return r[2] > surface_cutoff_;
        }

    private:
        jams::Real surface_cutoff_;
        jams::Real temporal_center_;
        jams::Real temporal_width_;
        Vec3R field_;
        Vec3R cache_field_ = {0.0, 0.0, 0.0};
        jams::Real cache_time_ = std::numeric_limits<jams::Real>::quiet_NaN();
    };


    class SincPulse : public TemporalFieldPulse {
    public:
        SincPulse(
            const jams::Real& temporal_width,
            const jams::Real& temporal_center, const Vec3R& field) :
            temporal_center_(temporal_center),
            temporal_width_(temporal_width),
            field_(field) {};

        explicit SincPulse(const libconfig::Setting &settings)
            : SincPulse(
            jams::config_required<jams::Real>(settings, "temporal_width"),
            jams::config_required<jams::Real>(settings, "temporal_center"),
            jams::config_required<Vec3R>(settings, "field")) {};

        Vec3R local_field(const jams::Real& time, const Vec3R& r) override {
          return max_field(time);
        }

        // Returns the maximum field at a given time and internally caches the
        // result. This avoid constant recalculation of the gaussian function
        // for function calls all at the same timestep.
        Vec3R max_field(const jams::Real& time) override {
          if (time == cache_time_) {
            return cache_field_;
          }

          cache_time_ = time;

          jams::Real x = temporal_width_ * (time - temporal_center_);
          cache_field_ = (sin(x) / x) * field_;

          return cache_field_;
        }

    private:
        jams::Real temporal_center_;
        jams::Real temporal_width_;
        Vec3R field_;
        Vec3R cache_field_ = {0.0, 0.0, 0.0};
        jams::Real cache_time_ = std::numeric_limits<jams::Real>::quiet_NaN();
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

Vec3R FieldPulseHamiltonian::calculate_field(int i, jams::Real time) {
  Vec3R r {globals::positions(i, 0), globals::positions(i, 1), globals::positions(i, 2)};
  return globals::mus(i) *
      temporal_field_pulse_->local_field(time, r);
}

jams::Real FieldPulseHamiltonian::calculate_energy(int i, jams::Real time) {
  Vec3R spin = array_cast<jams::Real>(Vec3{globals::s(i,0), globals::s(i, 1), globals::s(i, 2)});
  Vec3R field = calculate_field(i, time);
  return -dot(spin, field);
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
