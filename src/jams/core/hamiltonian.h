// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <iosfwd>
#include <cassert>

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/core/base.h"
#include "jams/interface/config.h"

class Hamiltonian : public Base {
public:
    Hamiltonian(const libconfig::Setting &settings, unsigned int size);

    virtual ~Hamiltonian() = default;

    // factory to create a Hamiltonian from a libconfig::Setting
    static Hamiltonian *create(const libconfig::Setting &settings, unsigned int size);

    // calculate the total energy of this Hamiltonian term
    virtual double calculate_total_energy() = 0;

    // calculate the energy of each spin and store in energy_
    virtual void calculate_energies() = 0;

    // calculate the field at each spin and store in field_
    virtual void calculate_fields() = 0;

    // calculate the field a spin i
    virtual Vec3 calculate_one_spin_field(int i) = 0;

    // calculate the energy of spin i
    virtual double calculate_one_spin_energy(int i) = 0;

    // calculate the energy difference of spin i in initial and final states
    virtual double calculate_one_spin_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) = 0;

    inline double energy(const int i) const {
      assert(i < energy_.elements());
      return energy_(i);
    }

    inline double field(const int i, const int j) const {
      assert(i < field_.size(0));
      assert(j < 3);
      return field_(i, j);
    }

    // raw pointer to field data on cuda device
    static double *dev_ptr_field() {
      #if HAS_CUDA
      return field_.device_data();
      #else
      return nullptr;
      #endif
    }

    // raw pointer to field data
    double *ptr_field() {
      return field_.data();
    }

protected:
    std::string name_; // Hamiltonian name
    std::string input_unit_name_; // name of energy unit in input
    double input_unit_conversion_ = 1.0; // conversion factor from input energy to JAMS native units

    jams::MultiArray<double, 1> energy_; // energy of every spin for this Hamiltonian
    jams::MultiArray<double, 2> field_; // field at every spin for this Hamiltonian
};


// Helper function to locate a chosen derived Hamiltonian within a list
// of Hamiltonian base classes
template<typename T>
const T *find_hamiltonian(const std::vector<Hamiltonian *> &hamiltonians) {
  for (const auto *ham : hamiltonians) {
    if (is_castable<const T *>(ham)) {
      return dynamic_cast<const T *>(ham);
    }
  }
  throw std::runtime_error("cannot find hamiltonian");
}

#endif  // JAMS_CORE_HAMILTONIAN_H
