// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>
#include <utility>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/units.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include "jams/hamiltonian/applied_field.h"
#include "jams/hamiltonian/anisotropy_polynomial.h"
#include "jams/hamiltonian/cubic_anisotropy.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/hamiltonian/exchange_neartree.h"
#include "jams/hamiltonian/exchange_functional.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/hamiltonian/uniaxial_anisotropy.h"
#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"
#include "jams/hamiltonian/zeeman.h"
#include "jams/hamiltonian/dipole_bruteforce.h"
#include "jams/hamiltonian/dipole_neartree.h"
#include "jams/hamiltonian/dipole_neighbour_list.h"
#include "jams/hamiltonian/dipole_fft.h"
#include "jams/hamiltonian/dipole_tensor.h"
#include "jams/hamiltonian/field_pulse.h"
#include "jams/hamiltonian/crystal_field.h"

#if HAS_CUDA
  #include "jams/hamiltonian/cuda_anisotropy_polynomial.h"
  #include "jams/hamiltonian/cuda_applied_field.h"
  #include "jams/hamiltonian/cuda_biquadratic_exchange.h"
  #include "jams/hamiltonian/cuda_cubic_anisotropy.h"
  #include "jams/hamiltonian/cuda_random_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h"
  #include "jams/hamiltonian/cuda_zeeman.h"
  #include "jams/hamiltonian/cuda_landau.h"
  #include "jams/hamiltonian/cuda_dipole_bruteforce.h"
  #include "jams/hamiltonian/cuda_dipole_fft.h"
  #include "jams/hamiltonian/cuda_field_pulse.h"
  #include "jams/hamiltonian/cuda_crystal_field.h"
#endif

#define DEFINED_HAMILTONIAN(name, type, settings, size) \
  { \
    if (lowercase(jams::config_required<std::string>(settings, "module")) == name) { \
      return new type(settings, size); \
    } \
  }

#ifdef HAS_CUDA
#define DEFINED_CUDA_HAMILTONIAN(name, type, settings, size) \
  { \
    if (lowercase(jams::config_required<std::string>(settings, "module")) == name) { \
      return new type(settings, size); \
    } \
  }
#else
#define DEFINED_CUDA_HAMILTONIAN(name, type, settings, size)
#endif

#ifdef HAS_CUDA
#define CUDA_HAMILTONIAN_NAME(type) Cuda##type
  #define DEFINED_HAMILTONIAN_CUDA_VARIANT(name, type, is_cuda_solver, settings, size) \
  { \
    if (lowercase(jams::config_required<std::string>(settings, "module")) == name) { \
      if(is_cuda_solver) { \
        return new CUDA_HAMILTONIAN_NAME(type)(settings, size); \
      } \
      return new type(settings, size); \
    } \
  }
#else
#define DEFINED_HAMILTONIAN_CUDA_VARIANT(name, type, is_cuda_solver, settings, size) \
  DEFINED_HAMILTONIAN(name, type, settings, size)
#endif

Hamiltonian * Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size, bool is_cuda_solver) {

  if (settings.exists("strategy")) {
    throw jams::removed_feature_error("dipole hamiltonians now have specific names and 'strategy' has been removed");
  }

  if (lowercase(jams::config_required<std::string>(settings, "module")) == "exchange") {
    return new ExchangeHamiltonian(settings, size, is_cuda_solver);
  }
  if (lowercase(jams::config_required<std::string>(settings, "module")) == "exchange-stencil") {
    throw jams::removed_feature_error(
        "exchange-stencil has been removed; use module = \"exchange\" with backend = \"stencil\"");
  }
  DEFINED_HAMILTONIAN("exchange-functional", ExchangeFunctionalHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange-neartree", ExchangeNeartreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-tensor", DipoleTensorHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neartree", DipoleNearTreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neighbour-list", DipoleNeighbourListHamiltonian, settings, size);

  DEFINED_CUDA_HAMILTONIAN("landau", CudaLandauHamiltonian, settings, size);
  DEFINED_CUDA_HAMILTONIAN("biquadratic-exchange", CudaBiquadraticExchangeHamiltonian, settings, size);

  DEFINED_HAMILTONIAN_CUDA_VARIANT("anisotropy-polynomial", AnisotropyPolynomialHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("applied-field", AppliedFieldHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("crystal-field", CrystalFieldHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("random-anisotropy", RandomAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("cubic-anisotropy", CubicAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial-anisotropy", UniaxialAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial-micro-anisotropy", UniaxialMicroscopicAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("zeeman", ZeemanHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("dipole-fft", DipoleFFTHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("dipole-bruteforce", DipoleBruteforceHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("field-pulse", FieldPulseHamiltonian, is_cuda_solver, settings, size);

  // Old names retained for compatibility
  DEFINED_HAMILTONIAN_CUDA_VARIANT("cubic", CubicAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial", UniaxialAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial-micro", UniaxialMicroscopicAnisotropyHamiltonian, is_cuda_solver, settings, size);



  throw std::runtime_error("unknown hamiltonian " + jams::config_required<std::string>(settings, "module"));
}

void Hamiltonian::calculate_fields(jams::Real time, const SpinArray& spins)
{
  const auto spin_view = spins.host_view();
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto local_field = calculate_field_from_spins(i, time, spin_view);
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = local_field[j];
    }
  }
}

bool Hamiltonian::supports_calculate_fields_in_parallel() const
{
  return false;
}

void Hamiltonian::calculate_fields_in_parallel(jams::Real, const SpinArray&)
{
  throw jams::unimplemented_error("Hamiltonian::calculate_fields_in_parallel");
}

void Hamiltonian::calculate_energies(jams::Real time, const SpinArray& spins)
{
  const auto spin_view = spins.host_view();
  for (auto i = 0; i < globals::num_spins; ++i) {
    energy_(i) = calculate_energy_from_spins(i, time, spin_view);
  }
}

jams::Real Hamiltonian::calculate_total_energy(jams::Real time, const SpinArray& spins)
{
  double e_total = 0.0;
  calculate_energies(time, spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    e_total += energy_(i);
  }
  return e_total;
}

jams::Real Hamiltonian::calculate_energy_difference(int i, const jams::Vec<double, 3>& spin_initial, const jams::Vec<double, 3>& spin_final,
  jams::Real time)
{
  const jams::Real e_initial = calculate_energy_for_spin(i, spin_initial, time);
  const jams::Real e_final = calculate_energy_for_spin(i, spin_final, time);

  return (e_final - e_initial);
}

Hamiltonian::EnergyCurrentInteractionSupport Hamiltonian::energy_current_interaction_support() const {
  return EnergyCurrentInteractionSupport::Unsupported;
}

void Hamiltonian::add_energy_current_interactions(jams::EnergyCurrentInteractionSink&) const {
}

jams::Vec<jams::Real, 3> Hamiltonian::calculate_field_from_spins(
    int i,
    jams::Real time,
    const SpinHostView&)
{
  return calculate_field(i, time);
}

jams::Real Hamiltonian::calculate_energy_from_spins(
    int i,
    jams::Real time,
    const SpinHostView&)
{
  return calculate_energy(i, time);
}

const Hamiltonian::SpinArray& Hamiltonian::global_spin_array_for_fields()
{
#if DO_MIXED_PRECISION
  if (fallback_spin_array_.elements() != globals::s.elements()) {
    fallback_spin_array_.resize(globals::s.extent(0), globals::s.extent(1));
  }

  const auto global_spins = std::as_const(globals::s).host_span();
  auto fallback_spins = fallback_spin_array_.mutable_host_span();
  std::transform(
      global_spins.begin(),
      global_spins.end(),
      fallback_spins.begin(),
      [](const double value) {
        return static_cast<jams::Real>(value);
      });
  return fallback_spin_array_;
#else
  return globals::s;
#endif
}

jams::Real Hamiltonian::calculate_energy_for_spin(int i, const jams::Vec<double, 3>& spin, jams::Real time)
{
  throw jams::unimplemented_error("Hamiltonian::calculate_energy_for_spin");
}

Hamiltonian::Hamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Base(settings),
          energy_(size),
          field_(size, 3)
{

  input_energy_unit_name_ = jams::config_optional<std::string>(settings, "energy_units", jams::defaults::energy_unit_name);

  // old setting name for backwards compatibility
  if (settings.exists("unit_name")) {
    input_energy_unit_name_ = jams::config_optional<std::string>(settings, "unit_name", jams::defaults::energy_unit_name);
  }

  if (!jams::internal_energy_unit_conversion.count(input_energy_unit_name_)) {
    throw std::runtime_error("energy units: " + input_energy_unit_name_ + " is not known");
  }

  input_energy_unit_conversion_ = jams::internal_energy_unit_conversion.at(input_energy_unit_name_);

  // global lattice must have been created before accessing ::lattice->parameter()
  assert(::globals::lattice);

  const std::map<std::string, double> internal_distance_unit_conversion = {
      {"lattice_constants", 1.0},
      {"m", 1.0 / ::globals::lattice->parameter()},
      {"meters", 1.0 / ::globals::lattice->parameter()},
      {"nm", 1e-9 / (::globals::lattice->parameter())}, // lattice parameter from config is in meters
      {"nanometers", 1e-9 / (::globals::lattice->parameter())},
      {"A", 1e-10 / (::globals::lattice->parameter() * 1e10)},
      {"angstroms", 1e-10 / (::globals::lattice->parameter())}
  };

  input_distance_unit_name_ = jams::config_optional<std::string>(settings, "distance_units", jams::defaults::distance_unit_name);

  if (!internal_distance_unit_conversion.count(input_distance_unit_name_)) {
    throw std::runtime_error("distance units: " + input_distance_unit_name_ + " is not known");
  }

  input_distance_unit_conversion_ = internal_distance_unit_conversion.at(input_distance_unit_name_);

  set_name(jams::config_required<std::string>(settings, "module"));
  std::cout << "  " << name() << " hamiltonian\n";


#ifdef HAS_CUDA
  cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
#endif

}
