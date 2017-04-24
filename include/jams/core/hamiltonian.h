// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <iosfwd>
#include <cassert>

#include "jblib/containers/array.h"

#ifdef CUDA
#include <cuda_runtime_api.h>
#include "jblib/containers/cuda_array.h"
#endif  // CUDA

#include "jams/core/types.h"
#include "jams/core/output.h"

// forward declarations
namespace libconfig {
  class Setting;
};

class Hamiltonian {
 public:
  Hamiltonian(const libconfig::Setting &settings, const unsigned int size) 
  : energy_(size, 0.0),
    field_(size, 3, 0.0)
  {}

  virtual ~Hamiltonian() {}

  // factory
  static Hamiltonian* create(const libconfig::Setting &settings, const unsigned int size);

  virtual std::string name() const = 0;

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
    #ifdef CUDA
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
    #ifdef CUDA
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
  jblib::Array<double, 1> energy_;
  jblib::Array<double, 2> field_;

#ifdef CUDA
  jblib::CudaArray<double, 1> dev_energy_;
  jblib::CudaArray<double, 1> dev_field_;
#endif  // CUDA
};

#endif  // JAMS_CORE_HAMILTONIAN_H