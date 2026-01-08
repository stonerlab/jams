// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HAMILTONIAN_H
#define JAMS_CORE_HAMILTONIAN_H

#include <jams/containers/multiarray.h>
#include <jams/core/base.h>
#include <jams/core/types.h>

#include <cassert>
#include <stdexcept>


#if HAS_CUDA
#include <cuda_runtime.h>
#include "jams/cuda/cuda_stream.h"
#endif

namespace libconfig { class Setting; }

class Hamiltonian : public Base {
public:
    Hamiltonian(const libconfig::Setting &settings, unsigned int size);

    Hamiltonian() = default;

    virtual ~Hamiltonian() = default;

    // factory to create a Hamiltonian from a libconfig::Setting
    static Hamiltonian *create(const libconfig::Setting &settings, unsigned int size, bool is_cuda_solver);

    // calculate the field a spin i
    virtual Vec3R calculate_field(int i, jams::Real time) = 0;

    // calculate the energy of spin i
    virtual jams::Real calculate_energy(int i, jams::Real time) = 0;

    // calculate the field at each spin and store in field_
    virtual void calculate_fields(jams::Real time);

    // calculate the energy of each spin and store in energy_
    virtual void calculate_energies(jams::Real time);

    // calculate the total energy of this Hamiltonian term
    virtual jams::Real calculate_total_energy(jams::Real time);

    // calculate the energy difference of spin i in initial and final states
    virtual jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time);


    inline jams::Real energy(const int i) const {
      assert(i < energy_.elements());
      return energy_(i);
    }

    inline jams::Real field(const int i, const int j) const {
      assert(i < field_.size(0));
      assert(j < 3);
      return field_(i, j);
    }

    // raw pointer to field data on cuda device
    jams::Real *dev_ptr_field() {
      #if HAS_CUDA
      return field_.device_data();
      #else
      return nullptr;
      #endif
    }

    // raw pointer to field data
    jams::Real *ptr_field() {
      return field_.data();
    }

#if HAS_CUDA

  cudaStream_t& get_stream()
    {
      return cuda_stream_.get();
    }

  // Call after enqueueing work to update the completion marker.
    void record_done()
    {
      cudaEventRecord(done_, cuda_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

  // Make an external stream wait for this Hamiltonian's work.
  void wait_on(cudaStream_t external) const {
      cudaStreamWaitEvent(external, done_, 0);
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

  // Host-side wait.
  void synchronize_done() const {
      cudaEventSynchronize(done_);
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }
#endif

protected:
    std::string input_energy_unit_name_; // name of energy unit in input
    double input_energy_unit_conversion_ = 1.0; // conversion factor from input energy to JAMS native units

    std::string input_distance_unit_name_; // name of distance unit in input
    double input_distance_unit_conversion_ = 1.0; // conversion factor from input distance to JAMS native units

    jams::MultiArray<jams::Real, 1> energy_; // energy of every spin for this Hamiltonian
    jams::MultiArray<jams::Real, 2> field_; // field at every spin for this Hamiltonianl

#if HAS_CUDA
  CudaStream cuda_stream_ {};
  cudaEvent_t  done_{};
#endif
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
