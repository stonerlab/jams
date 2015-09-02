// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_DIPOLE_EWALD_H
#define JAMS_HAMILTONIAN_DIPOLE_EWALD_H

#include "hamiltonian/strategy.h"
#include "core/consts.h"
#include "core/maths.h"


class DipoleHamiltonianEwald : public HamiltonianStrategy {
    public:
        DipoleHamiltonianEwald(const libconfig::Setting &settings);

        ~DipoleHamiltonianEwald() {};

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) ;
        void   calculate_energies(jblib::Array<double, 1>& energies);

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields(jblib::Array<double, 2>& fields);
    private:
        inline double fG(const double r, const double a);
        inline double pG(const double r, const double a);

        void   calculate_local_ewald_field(const int i, double h[3]);
        void   calculate_self_ewald_field(const int i, double h[3]);
        void   calculate_surface_ewald_field(const int i, double h[3]);
        void   calculate_nonlocal_ewald_field();

        double surf_elec_;
        double sigma_;
        double delta_error_;
        double r_cutoff_;
        int k_cutoff_;

        SparseMatrix<double>          local_interaction_matrix_;

        jblib::Vec3<int> kspace_size_;
        jblib::Vec3<int> kspace_padded_size_;

        jblib::Array<double, 4> h_nonlocal_;
        jblib::Array<double, 4> s_nonlocal_;
        jblib::Array<double, 2> s_old_1_;
        jblib::Array<double, 2> s_old_2_;
        jblib::Array<double, 4> h_nonlocal_2_;


        jblib::Array<fftw_complex, 4> s_recip_;
        jblib::Array<fftw_complex, 4> h_recip_;
        jblib::Array<double, 5> w_recip_;

        fftw_plan spin_fft_forward_transform_;
        fftw_plan field_fft_backward_transform_;
};

inline double DipoleHamiltonianEwald::fG(const double r, const double a) {
    return erfc(r / (a*sqrt(2.0))) + ((kSqrtTwo*r)/(kPi*a))*exp(-0.5*pow(r/a, 2));
}

// Gaussian point dipole
inline double DipoleHamiltonianEwald::pG(const double r, const double a) {
    return sqrt(2.0/kPi)*exp(-0.5*pow(r/a, 2))/(pow(r, 2)*pow(a, 3));
}

#endif  // JAMS_HAMILTONIAN_DIPOLE_EWALD_H