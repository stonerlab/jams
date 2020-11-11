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

#if HAS_CUDA
  #include "jams/hamiltonian/cuda_cubic_anisotropy.h"
  #include "jams/hamiltonian/cuda_random_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_anisotropy.h"
  #include "jams/hamiltonian/cuda_uniaxial_microscopic_anisotropy.h"
  #include "jams/hamiltonian/cuda_zeeman.h"
  #include "jams/hamiltonian/cuda_dipole_bruteforce.h"
  #include "jams/hamiltonian/cuda_dipole_fft.h"
#endif

#define DEFINED_HAMILTONIAN(name, type, settings, size) \
  { \
    if (lowercase(settings["module"]) == name) { \
      return new type(settings, size); \
    } \
  }

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


using namespace std;

Hamiltonian * Hamiltonian::create(const libconfig::Setting &settings, const unsigned int size, bool is_cuda_solver) {

  if (settings.exists("strategy")) {
    throw jams::removed_feature_error("dipole hamiltonians now have specific names and 'strategy' has been removed");
  }

  DEFINED_HAMILTONIAN("applied-field", AppliedFieldHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange", ExchangeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange-functional", ExchangeFunctionalHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("exchange-neartree", ExchangeNeartreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-tensor", DipoleTensorHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neartree", DipoleNearTreeHamiltonian, settings, size);
  DEFINED_HAMILTONIAN("dipole-neighbour-list", DipoleNeighbourListHamiltonian, settings, size);

  DEFINED_HAMILTONIAN_CUDA_VARIANT("random-anisotropy", RandomAnisotropyHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial", UniaxialHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("uniaxial-micro", UniaxialMicroscopicHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("cubic", CubicHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("zeeman", ZeemanHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("dipole-fft", DipoleFFTHamiltonian, is_cuda_solver, settings, size);
  DEFINED_HAMILTONIAN_CUDA_VARIANT("dipole-bruteforce", DipoleBruteforceHamiltonian, is_cuda_solver, settings, size);


  throw std::runtime_error("unknown hamiltonian " + std::string(settings["module"].c_str()));
}

Hamiltonian::Hamiltonian(const libconfig::Setting &settings, const unsigned int size)
        : Base(settings),
          energy_(size),
          field_(size, 3),
          input_unit_name_(jams::config_optional<string>(settings, "unit_name", jams::defaults::energy_unit_name))
{
  set_name(settings["module"].c_str());
  cout << "  " << name() << " hamiltonian\n";

  if (!jams::internal_energy_unit_conversion.count(input_unit_name_)) {
    throw runtime_error("units: " + input_unit_name_ + " is not known");
  }

  input_unit_conversion_ = jams::internal_energy_unit_conversion.at(input_unit_name_);
}
