// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <iosfwd>
#include <cassert>

#include "jblib/containers/array.h"

#if HAS_CUDA
#include <cuda_runtime_api.h>
#include "jblib/containers/cuda_array.h"
#endif  // CUDA

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/core/base.h"

// forward declarations
namespace libconfig {
  class Setting;
};

class Hamiltonian : public Base {
 public:
  Hamiltonian(const libconfig::Setting &settings, const unsigned int size);

  virtual ~Hamiltonian() = default;

  // factory
  static Hamiltonian *create(const libconfig::Setting &settings, const unsigned int size, bool is_cuda_solver);

  virtual double calculate_total_energy() = 0;
  virtual double calculate_one_spin_energy(const int i) = 0;
  virtual double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;

  virtual void   calculate_energies() = 0;

  virtual void   calculate_one_spin_field(const int i, double h[3]) = 0;
  virtual void   calculate_fields() = 0;

  inline double energy(const int i) const {
    assert(i < energy_.elements());
    return energy_(i);
  }

  inline double field(const int i, const int j) const {
    assert(i < field_.size(0));
    assert(j < 3);
    return field_(i,j);
  }

  double* dev_ptr_energy() {
    #if HAS_CUDA
    assert(dev_energy_.is_allocated());
    return dev_energy_.data();
    #else
    return NULL;
    #endif
  }
  double* ptr_energy() {
    assert(energy_.is_allocated());
    return energy_.data();
  }

  double* dev_ptr_field() {
    #if HAS_CUDA
    assert(dev_field_.is_allocated());
    return dev_field_.data();
    #else
    return NULL;
    #endif
  }
  double* ptr_field() {
    assert(field_.is_allocated());
    return field_.data();
  }

 protected:
    std::string name_;
    std::string input_unit_name_;
    double      input_unit_conversion_ = 1.0;

  jblib::Array<double, 1> energy_;
  jblib::Array<double, 2> field_;

#if HAS_CUDA
  jblib::CudaArray<double, 1> dev_energy_;
  jblib::CudaArray<double, 1> dev_field_;
#endif  // CUDA
};


// Helper function to locate a chosen derived Hamiltonian within a list
// of Hamiltonian base classes
template <typename T>
const T* find_hamiltonian(const std::vector<Hamiltonian*>&hamiltonians) {
  for (const auto* ham : hamiltonians) {
    if (is_castable<const T*>(ham)) {
      return dynamic_cast<const T*>(ham);
    }
  }
  throw std::runtime_error("cannot find hamiltonian");
}

#endif  // JAMS_CORE_HAMILTONIAN_H
