// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
#define JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H

#include <jams/core/types.h>
#include <jams/core/hamiltonian.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/helpers/mixed_precision.h>

#include <array>

#include <cufft.h>

class CudaDipoleFFTHamiltonian : public Hamiltonian {
    public:
        CudaDipoleFFTHamiltonian(const libconfig::Setting &settings, unsigned int size);
        ~CudaDipoleFFTHamiltonian() override;

        double calculate_total_energy(double time) override;
        double calculate_energy(int i, double time) override;
        double calculate_one_spin_energy(int i, const Vec3 &s_i, double time);
        double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override ;
        void   calculate_energies(double time) override;

        Vec3   calculate_field(int i, double time);
        void   calculate_fields(double time) override;
    private:
        bool debug_ = false;
        bool check_radius_   = true;
        bool check_symmetry_ = true;

        jams::MultiArray<jams::ComplexLo, 4> generate_kspace_dipole_tensor(const int pos_i, const int pos_j, std::vector<Vec3> &generated_positions);

        double                          r_cutoff_;
        double                          distance_tolerance_;


        Vec3i                    kspace_size_;
        Vec3i                    kspace_padded_size_;

        jams::MultiArray<float, 2> s_float_;
        jams::MultiArray<float, 2> h_float_;
        jams::MultiArray<float, 1> mus_float_;


        jams::MultiArray<jams::cufftComplexLo, 1>   kspace_s_;
        jams::MultiArray<jams::cufftComplexLo, 1>   kspace_h_;

        CudaStream dev_stream_;

        // size is num_sites, num_sites, num_tensor_components, num_kpoints
        jams::MultiArray<jams::cufftComplexLo, 3> kspace_tensors_;

        cufftHandle                     cuda_fft_s_rspace_to_kspace;
        cufftHandle                     cuda_fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
