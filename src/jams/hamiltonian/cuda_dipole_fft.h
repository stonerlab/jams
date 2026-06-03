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
        jams::Real calculate_one_spin_energy(int i, const jams::Vec<double, 3> &s_i, jams::Real time);
        jams::Real calculate_energy_difference(int i, const jams::Vec<double, 3> &spin_initial, const jams::Vec<double, 3> &spin_final, jams::Real time) override ;
        void   calculate_energies(jams::Real time) override;

        jams::Vec<jams::Real, 3>   calculate_field(int i, jams::Real time);
        void   calculate_fields(jams::Real time) override;

        void add_energy_current_interactions(jams::SparseMatrix<double>::Builder& rx_builder,
                                             jams::SparseMatrix<double>::Builder& ry_builder,
                                             jams::SparseMatrix<double>::Builder& rz_builder) const override;
    private:
        bool debug_ = false;
        bool check_radius_   = true;
        bool check_symmetry_ = true;

        void generate_kspace_dipole_tensor(const int pos_i, const int pos_j, const int pair, std::vector<jams::Vec<double, 3>> &generated_positions);

        jams::Real                          r_cutoff_;
        jams::Real                          distance_tolerance_;


        jams::Vec<int, 3>                    kspace_size_;
        jams::Vec<int, 3>                    kspace_padded_size_;

        bool use_dense_fft_buffers_ = false;
        bool use_full_tensor_storage_ = false;
        jams::MultiArray<int, 1> fft_site_map_;
        jams::MultiArray<jams::Real, 1> rspace_s_dense_;
        jams::MultiArray<jams::Real, 1> rspace_h_dense_;

        jams::MultiArray<jams::Real, 2> s_float_;
        jams::MultiArray<jams::cufftComplex, 1>   kspace_s_;
        jams::MultiArray<jams::cufftComplex, 1>   kspace_h_;

        // compact path: upper-triangular motif pairs
        // padded/cropped path: full motif-pair table
        jams::MultiArray<jams::cufftComplex, 3> kspace_tensors_;

        cufftHandle                     cuda_fft_s_rspace_to_kspace;
        cufftHandle                     cuda_fft_h_kspace_to_rspace;
};

#endif  // JAMS_HAMILTONIAN_CUDA_DIPOLE_FFT_H
