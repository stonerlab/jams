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

      void   calculate_energies(double time) override;
      void   calculate_fields(double time) override;
      double calculate_total_energy(double time) override;

      Vec3   calculate_field(const int i, double time) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      double calculate_energy(const int i, double time) final {
        JAMS_UNIMPLEMENTED_FUNCTION; }
      double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) final {
        JAMS_UNIMPLEMENTED_FUNCTION;}

    private:
      unsigned   dev_blocksize_ = 128;

      CudaStream dev_stream_;
      thrust::device_vector<double> dev_magnitude_;
      thrust::device_vector<double> dev_direction_;
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_CUDA_H