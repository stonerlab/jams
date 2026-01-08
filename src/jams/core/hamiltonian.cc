// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>

#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/units.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"

#include "jams/hamiltonian/applied_field.h"
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
    if (lowercase(settings["module"]) == name) { \
      return new type(settings, size); \
    } \
  }

#ifdef HAS_CUDA
#define DEFINED_CUDA_HAMILTONIAN(name, type, settings, size) \
  { \
    if (lowercase(settings["module"]) == name) { \
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
    if (lowercase(settings["module"]) == name) { \
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

  DEFINED_HAMILTONIAN("exchange", ExchangeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange-functional", ExchangeFunctionalHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange-neartree", ExchangeNeartreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-tensor", DipoleTensorHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neartree", DipoleNearTreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neighbour-list", DipoleNeighbourListHamiltonian, settings, size);

  DEFINED_CUDA_HAMILTONIAN("landau", CudaLandauHamiltonian, settings, size);
  DEFINED_CUDA_HAMILTONIAN("biquadratic-exchange", CudaBiquadraticExchangeHamiltonian, settings, size);

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



  throw std::runtime_error("unknown hamiltonian " + std::string(settings["module"].c_str()));
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

  set_name(settings["module"].c_str());
  std::cout << "  " << name() << " hamiltonian\n";


#ifdef HAS_CUDA
  cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
#endif

}
