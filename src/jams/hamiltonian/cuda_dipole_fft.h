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

        jams::Real calculate_total_energy(jams::Real time) override;
        jams::Real calculate_energy(int i, jams::Real time) override;
        jams::Real calculate_one_spin_energy(int i, const Vec3 &s_i, jams::Real time);
        jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override ;
        void   calculate_energies(jams::Real time) override;

        Vec3R   calculate_field(int i, jams::Real time);
        void   calculate_fields(jams::Real time) override;
    private:
        bool debug_ = false;
        bool check_radius_   = true;
        bool check_symmetry_ = true;

        void generate_kspace_dipole_tensor(const int pos_i, const int pos_j, const int pair, std::vector<Vec3> &generated_positions);

        jams::Real                          r_cutoff_;
        jams::Real                          distance_tolerance_;


        Vec3i                    kspace_size_;
        Vec3i                    kspace_padded_size_;

        jams::MultiArray<jams::Real, 2> s_float_;
        jams::MultiArray<jams::Real, 1> mus_unitcell_;


        jams::MultiArray<jams::cufftComplex, 1>   kspace_s_;
        jams::MultiArray<jams::cufftComplex, 1>   kspace_h_;

        // size is num_sites, num_sites, num_tensor_components, num_kpoints
        jams::MultiArray<jams::cufftComplex, 3> kspace_tensors_;

        cufftHandle                     cuda_fft_s_rspace_to_kspace;
        cufftHandle                     cuda_fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
