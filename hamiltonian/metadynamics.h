// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_HAMILTONIAN_METADYNAMICS_H
#define JAMS_HAMILTONIAN_METADYNAMICS_H

#include <libconfig.h++>
#include <array>

#include "core/output.h"
#include "core/hamiltonian.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class MetadynamicsHamiltonian : public Hamiltonian {
    public:
        friend class MetadynamicsPhysics;

        MetadynamicsHamiltonian(const libconfig::Setting &settings);
        ~MetadynamicsHamiltonian() {};

        std::string name() const { return "METADYNAMICS"; }

        double calculate_total_energy();
        double calculate_one_spin_energy(const int i);
        double calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final);

        double calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final);

        void   calculate_energies();

        void   calculate_one_spin_field(const int i, double h[3]);
        void   calculate_fields();

        void   output_energies(OutputFormat format);
        void   output_fields(OutputFormat format);

    protected:
        void   add_gaussian();
        void   calculate_collective_variables();
        void output_gaussians(std::ostream &out);
        double gaussian(double x, double y);
    private:



        void output_energies_text();
        // void output_energies_hdf5();

        void output_fields_text();
        // void output_fields_hdf5();

        double                 cv_mag_x;
        double                 cv_mag_y;
        double                 cv_mag_z;

        double                 cv_mag_t;
        double                 cv_mag_p;

        jblib::Array<double, 2> collective_variable_deriv;

        std::vector<std::array<double, 2>> gaussian_centers;   
        double             gaussian_width;
        double             gaussian_height;
        int             gaussian_placement_interval;

#ifdef CUDA
        cudaStream_t dev_stream_;
#endif  // CUDA

};

#endif  // JAMS_HAMILTONIAN_METADYNAMICS_H