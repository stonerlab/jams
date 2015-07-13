// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <string>

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

#include "core/output.h"

// forward declarations
namespace libconfig {
  class Setting;
};

namespace jblib {
  template <typename T>
  class Vec3;
};

class Hamiltonian {
 public:
  Hamiltonian(const libconfig::Setting &settings) {}

  virtual ~Hamiltonian() {}

  // factory
  static Hamiltonian* create(const libconfig::Setting &settings);

  virtual double calculate_total_energy() = 0;
  virtual double calculate_one_spin_energy(const int i) = 0;
  virtual         double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) = 0;

  virtual void   calculate_energies() = 0;

  virtual void   calculate_one_spin_fields(const int i, double h[3]) = 0;
  virtual void   calculate_fields() = 0;

  virtual void   output_energies(OutputFormat format) = 0;
  virtual void   output_fields(OutputFormat format) = 0;

  double* dev_ptr_energy() {
    assert(dev_energy_.is_allocated());
    return dev_energy_.data();
  }
  double* ptr_energy() {
    assert(energy_.is_allocated());
    return energy_.data();
  }

  double* dev_ptr_field() {
    assert(dev_field_.is_allocated());
    return dev_field_.data();
  }
  double* ptr_field() {
    assert(field_.is_allocated());
    return field_.data();
  }

 protected:

  virtual void output_energies_text() = 0;
  // virtual void output_energies_hdf5() = 0;

  virtual void output_fields_text() = 0;
  // virtual void output_fields_hdf5() = 0;


  OutputFormat            outformat_;
  jblib::Array<double, 1> energy_;
  jblib::Array<double, 2> field_;

#ifdef CUDA
  jblib::CudaArray<double, 1> dev_energy_;
  jblib::CudaArray<double, 1> dev_field_;
#endif  // CUDA
};

#endif  // JAMS_CORE_HAMILTONIAN_H
