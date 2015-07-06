// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <string>

#include "jblib/containers/array.h"

class Hamiltonian {
 public:
  Hamiltonian(const libconfig::Setting &settings) {}

  virtual ~Hamiltonian() {}

  // factory
  static Hamiltonian* create(const libconfig::Setting &settings);

  virtual double calculate_total_energy() = 0;
  virtual double calculate_one_spin_energy(const int i) = 0;
  virtual void   calculate_energies() = 0;

  virtual void   calculate_one_spin_fields(const int i, double h[3]) = 0;
  virtual void   calculate_fields() = 0;

  virtual void   output_energies(OutputFormat format) = 0;
  virtual void   output_fields(OutputFormat format) = 0;

 private:

  virtual void output_energies_text() = 0;
  // virtual void output_energies_hdf5() = 0;

  virtual void output_fields_text() = 0;
  // virtual void output_fields_hdf5() = 0;

};

#endif  // JAMS_CORE_HAMILTONIAN_H
