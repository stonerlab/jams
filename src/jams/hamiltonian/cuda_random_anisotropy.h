//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H

#include <jams/hamiltonian/random_anisotropy.h>
#include <jams/helpers/exception.h>
#include <jams/cuda/cuda_stream.h>

#include <vector>

#include <thrust/device_vector.h>

class CudaRandomAnisotropyHamiltonian : public RandomAnisotropyHamiltonian {
    public:
      CudaRandomAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);
      ~CudaRandomAnisotropyHamiltonian() override = default;

      void   calculate_energies(jams::Real time) override;
      void   calculate_fields(jams::Real time) override;
      jams::Real calculate_total_energy(jams::Real time) override;

      Vec3R   calculate_field(const int i, jams::Real time) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      jams::Real calculate_energy(const int i, jams::Real time) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) final {
        JAMS_UNIMPLEMENTED_FUNCTION;}

    private:
      unsigned   dev_blocksize_ = 128;
      thrust::device_vector<jams::Real> dev_magnitude_;
      thrust::device_vector<jams::Real> dev_direction_;
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H