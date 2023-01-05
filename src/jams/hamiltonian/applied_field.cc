
#include <jams/hamiltonian/applied_field.h>
#include <jams/containers/vec3.h>
#include <jams/core/globals.h>
#include <jams/interface/config.h>

#include <iostream>

namespace {

struct StaticField : public AppliedFieldHamiltonian::TimeDependentField {
public:
    explicit StaticField(const libconfig::Setting &settings)
    : b_field_(jams::config_required<Vec3>(settings, "field"))
    {}

    Vec3 field(const double time) override {
      return b_field_;
    }
private:
    Vec3 b_field_ = {0, 0, 0};
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
        : b_field_(jams::config_required<Vec3>(settings, "field"))
        , time_center_(jams::config_required<double>(settings, "time_center") / 1e-12)
        , freq_bandwidth_(jams::config_required<double>(settings, "freq_bandwidth")  / 1e12) //  Thz
    {}

    Vec3 field(const double time) override {
      return b_field_ * sinc(kPi * freq_bandwidth_ * (time - time_center_));
    }
private:
    Vec3 b_field_ = {0, 0, 0};
    double time_center_ = 0.0;
    double freq_bandwidth_ = 0.0;
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
        : b_field_(jams::config_required<Vec3>(settings, "field"))
          , time_center_(jams::config_required<double>(settings, "time_center") / 1e-12)
          , freq_bandwidth_(jams::config_required<double>(settings, "freq_bandwidth")  / 1e12) //  THz
          , freq_center_(jams::config_required<double>(settings, "freq_center")  / 1e12) //  THz
    {}

    Vec3 field(const double time) override {
      return b_field_ * sinc(kPi * freq_bandwidth_ * (time - time_center_))
                      * cos(kTwoPi * freq_center_ * (time - time_center_));
    }
private:
    Vec3 b_field_ = {0, 0, 0};
    double time_center_ = 0.0;
    double freq_bandwidth_ = 0.0;
    double freq_center_ = 0.0;
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

double AppliedFieldHamiltonian::calculate_total_energy(double time) {
  double e_total = 0.0;
  calculate_energies(time);
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

void AppliedFieldHamiltonian::calculate_energies(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy(i, time);
  }
}

void AppliedFieldHamiltonian::calculate_fields(double time) {
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field(i, time);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

Vec3 AppliedFieldHamiltonian::calculate_field(int i, double time) {
  return globals::mus(i) * time_dependent_field_->field(time);
}

double AppliedFieldHamiltonian::calculate_energy(int i, double time) {
  auto field = calculate_field(i, time);
  return -( globals::s(i,0) * field[0]
          + globals::s(i,1) * field[1]
          + globals::s(i,2) * field[2]);
}

double AppliedFieldHamiltonian::calculate_energy_difference(int i,
                                                            const Vec3 &spin_initial,
                                                            const Vec3 &spin_final, double time) {
  const auto e_initial = -dot(spin_initial, calculate_field(i, time));
  const auto e_final = -dot(spin_final, calculate_field(i, time));

  return (e_final - e_initial);
}




