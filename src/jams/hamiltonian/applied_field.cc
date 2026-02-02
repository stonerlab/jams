
#include <jams/hamiltonian/applied_field.h>
#include <jams/containers/vec3.h>
#include <jams/core/globals.h>
#include <jams/interface/config.h>

#include <iostream>

namespace {

struct StaticField : public AppliedFieldHamiltonian::TimeDependentField {
public:
    explicit StaticField(const libconfig::Setting &settings)
    : b_field_(jams::config_required<Vec3R>(settings, "field"))
    {}

    Vec3R field(const jams::Real time) override {
      return b_field_;
    }
private:
    Vec3R b_field_ = {0, 0, 0};
};

/// Implements an AppliedFieldHamiltonian functional class for the field
///
/// B(t) = B * sinc(π * f_bw * (t - t0))
///
/// where f_bw is the frequency bandwidth and t0 is the time center. Note that
/// the frequency bandwidth is centered on zero, so a bandwidth of 10 GHz
/// excites frequencies from -5 GHz to 5 GHz.
///

struct SincField : public AppliedFieldHamiltonian::TimeDependentField {
public:
    explicit SincField(const libconfig::Setting &settings)
        : b_field_(jams::config_required<Vec3R>(settings, "field"))
        , time_center_(jams::config_required<jams::Real>(settings, "time_center") / 1e-12)
        , freq_bandwidth_(jams::config_required<jams::Real>(settings, "freq_bandwidth")  / 1e12) //  Thz
    {}

    Vec3R field(const jams::Real time) override {
      return b_field_ * static_cast<jams::Real>(jams::sinc(kPi * freq_bandwidth_ * (time - time_center_)));
    }
private:
    Vec3R b_field_ = {0, 0, 0};
    jams::Real time_center_ = 0.0;
    jams::Real freq_bandwidth_ = 0.0;
};

/// Implements an AppliedFieldHamiltonian functional class for the field
///
/// B(t) = B * sinc(π * f_bw * (t - t0)) * cos(2π * f_c * (t - t0))
///
/// where f_bw is the frequency bandwidth and t0 is the time center and f_c is
/// the frequency center. This allows discrete frequency ranges to be excited as
/// a top hat function in frequency space with a cutoff above and below.
///
/// See Träger, Sci. Rep. 10, 18146 (2020)
/// [https://doi.org/10.1038/s41598-020-74785-4] for example diagrams.
///
struct SincCosField : public AppliedFieldHamiltonian::TimeDependentField {
public:
    explicit SincCosField(const libconfig::Setting &settings)
        : b_field_(jams::config_required<Vec3R>(settings, "field"))
          , time_center_(jams::config_required<jams::Real>(settings, "time_center") / 1e-12)
          , freq_bandwidth_(jams::config_required<jams::Real>(settings, "freq_bandwidth")  / 1e12) //  THz
          , freq_center_(jams::config_required<jams::Real>(settings, "freq_center")  / 1e12) //  THz
    {}

    Vec3R field(const jams::Real time) override {
      return b_field_ * static_cast<jams::Real>(jams::sinc(kPi * freq_bandwidth_ * (time - time_center_))
                      * cos(kTwoPi * freq_center_ * (time - time_center_)));
    }
private:
    Vec3R b_field_ = {0, 0, 0};
    jams::Real time_center_ = 0.0;
    jams::Real freq_bandwidth_ = 0.0;
    jams::Real freq_center_ = 0.0;
};


}

AppliedFieldHamiltonian::AppliedFieldHamiltonian(
    const libconfig::Setting &settings, unsigned int size)
    : Hamiltonian(settings, size) {

  if (settings.exists("type")) {
    auto type = lowercase(jams::config_required<std::string>(settings, "type"));

    if (type == "static") {
      time_dependent_field_ = std::make_unique<StaticField>(settings);
    }
    else if (type == "sinc") {
      time_dependent_field_ = std::make_unique<SincField>(settings);
    }
    else if (type == "sinc-cos") {
        time_dependent_field_ = std::make_unique<SincCosField>(settings);
    } else {
      throw std::runtime_error("Unknown field pulse type " + type);
    }

    std::cout << "field type: " << type << std::endl;
    this->set_name(name() + "-" + type);
  } else {
    // Backwards compatibility with configs where the type was not
    // specified meaning that the field is a static applied field.
    time_dependent_field_ = std::make_unique<StaticField>(settings);

    std::cout << "field type: static" << std::endl;
    this->set_name(name() + "-static");
  }


  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = globals::mus(i) * time_dependent_field_->field(0.0)[j];
    }
  }
}



Vec3R AppliedFieldHamiltonian::calculate_field(int i, jams::Real time) {
  return globals::mus(i) * time_dependent_field_->field(time);
}

jams::Real AppliedFieldHamiltonian::calculate_energy(int i, jams::Real time) {
  auto field = calculate_field(i, time);
  return -( globals::s(i,0) * field[0]
          + globals::s(i,1) * field[1]
          + globals::s(i,2) * field[2]);
}




