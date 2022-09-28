//
// Created by Sean Stansill [ll14s26s] on 26/09/2022.
//

#ifndef JAMS_HELICITY_MODULUS_H
#define JAMS_HELICITY_MODULUS_H

#if HAS_CUDA
#include <cusparse.h>
#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_common.h"
#endif

#include <fstream>

#include <libconfig.h++>

#include "jams/containers/sparse_matrix.h"
#include "jams/core/types.h"
#include "jams/core/solver.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"

class HelicityModulusMonitor : public Monitor {
public:
    explicit HelicityModulusMonitor(const libconfig::Setting &settings);

    ~HelicityModulusMonitor() override = default;

    void update(Solver *solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;
    std::string   tsv_header();
    Mat3 exchange_total_internal_energy_difference();
    Mat3 exchange_total_entropy();
    void calculate_helicity_fields();
    void calculate_entropy_fields();
    jams::SparseMatrix<double> interaction_Jij_;
    jams::SparseMatrix<double> interaction_rij_x_;
    jams::SparseMatrix<double> interaction_rij_y_;
    jams::SparseMatrix<double> interaction_rij_z_;
    jams::MultiArray<double, 2> helicity_field_rxx_; // exchange helicity field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> helicity_field_rxy_; // exchange helicity field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> helicity_field_rxz_; // exchange helicity field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> helicity_field_ryy_; // exchange helicity field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> helicity_field_ryz_; // exchange helicity field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> helicity_field_rzz_; // exchange helicity field at every spin for this Hamiltonian

    jams::MultiArray<double, 2> entropy_field_x_; // exchange entropy field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> entropy_field_y_; // exchange entropy field at every spin for this Hamiltonian
    jams::MultiArray<double, 2> entropy_field_z_; // exchange entropy field at every spin for this Hamiltonian

    #if HAS_CUDA
        CudaStream cusparse_stream_; // cuda stream to run in
    #endif

};

#endif //JAMS_HELICITY_MODULUS_H