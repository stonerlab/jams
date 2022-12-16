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
    static Hamiltonian *create(const libconfig::Setting &settings, unsigned int size, bool is_cuda_solver);

    // calculate the total energy of this Hamiltonian term
    virtual double calculate_total_energy(double time) = 0;

    // calculate the energy of each spin and store in energy_
    virtual void calculate_energies(double time) = 0;

    // calculate the field at each spin and store in field_
    virtual void calculate_fields(double time) = 0;

    // calculate the field a spin i
    virtual Vec3 calculate_field(int i, double time) = 0;

    // calculate the energy of spin i
    virtual double calculate_energy(int i, double time) = 0;

    // calculate the energy difference of spin i in initial and final states
    virtual double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) = 0;

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
    double *dev_ptr_field() {
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
    std::string input_energy_unit_name_; // name of energy unit in input
    double input_energy_unit_conversion_ = 1.0; // conversion factor from input energy to JAMS native units

    std::string input_distance_unit_name_; // name of distance unit in input
    double input_distance_unit_conversion_ = 1.0; // conversion factor from input distance to JAMS native units

    jams::MultiArray<double, 1> energy_; // energy of every spin for this Hamiltonian
    jams::MultiArray<double, 2> field_; // field at every spin for this Hamiltonianl
};


// Helper function to locate a chosen derived Hamiltonian within a list
// of Hamiltonian base classes
template<typename T>
const T &find_hamiltonian(const std::vector<std::unique_ptr<Hamiltonian>> &hamiltonians) {
  for (const auto &ham : hamiltonians) {
    if (is_castable<const T*>(ham.get())) {
      return dynamic_cast<const T&>(*ham);
    }
  }
  throw std::runtime_error("cannot find hamiltonian");
}

#endif  // JAMS_CORE_HAMILTONIAN_H
