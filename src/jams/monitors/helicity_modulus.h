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
    jams::SparseMatrix<double> interaction_JRR_xx_;
    jams::SparseMatrix<double> interaction_JRR_xy_;
    jams::SparseMatrix<double> interaction_JRR_xz_;
    jams::SparseMatrix<double> interaction_JRR_yy_;
    jams::SparseMatrix<double> interaction_JRR_yz_;
    jams::SparseMatrix<double> interaction_JRR_zz_;

    jams::SparseMatrix<double> interaction_JR_x_;
    jams::SparseMatrix<double> interaction_JR_y_;
    jams::SparseMatrix<double> interaction_JR_z_;

    jams::MultiArray<double, 2> helicity_field_rxrx_;
    jams::MultiArray<double, 2> helicity_field_rxry_;
    jams::MultiArray<double, 2> helicity_field_rxrz_;

    jams::MultiArray<double, 2> helicity_field_ryry_;
    jams::MultiArray<double, 2> helicity_field_ryrz_;
    jams::MultiArray<double, 2> helicity_field_rzrz_;

    jams::MultiArray<double, 2> entropy_field_rx_;
    jams::MultiArray<double, 2> entropy_field_ry_;
    jams::MultiArray<double, 2> entropy_field_rz_;

    #if HAS_CUDA
        CudaStream cusparse_stream_; // cuda stream to run in
    #endif

};

#endif //JAMS_HELICITY_MODULUS_H