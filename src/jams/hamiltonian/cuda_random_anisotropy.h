//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"
#include "jams/hamiltonian/random_anisotropy.h"
#include "jams/helpers/exception.h"
#include "jams/cuda/cuda_stream.h"

#include <thrust/device_vector.h>
#include "jams/cuda/cuda_stream.h"

class CudaRandomAnisotropyHamiltonian : public RandomAnisotropyHamiltonian {
    public:
      CudaRandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);
      ~CudaRandomAnisotropyHamiltonian() override = default;

      void   calculate_energies() override;
      void   calculate_fields() override;
      double calculate_total_energy() override;

      void   calculate_one_spin_field(const int i, double h[3]) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      double calculate_one_spin_energy(const int i) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      double calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) final {
        JAMS_UNIMPLEMENTED_FUNCTION;}

    private:
      unsigned   dev_blocksize_ = 128;

      CudaStream dev_stream_;
      thrust::device_vector<double> dev_magnitude_;
      thrust::device_vector<double> dev_direction_;
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H