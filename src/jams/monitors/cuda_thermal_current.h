//
// Created by Joe Barker on 2018/04/20.
//

#ifndef JAMS_CUDA_THERMAL_CURRENT_H
#define JAMS_CUDA_THERMAL_CURRENT_H

#include <jams/containers/multiarray.h>
#include <jams/containers/sparse_matrix.h>
#include <jams/core/monitor.h>
#include <jams/core/types.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/helpers/output.h>

#include <string>
#include <vector>

jams::Vec<double, 3> execute_cuda_thermal_current_kernel(
    CudaStream &stream,
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<jams::Real, 2>& field,
    const jams::MultiArray<jams::Real, 1>& gyro,
    const jams::MultiArray<jams::Real, 1>& mus,
    jams::SparseMatrix<double>& energy_current_operator_rx,
    jams::SparseMatrix<double>& energy_current_operator_ry,
    jams::SparseMatrix<double>& energy_current_operator_rz,
    double current_density_prefactor,
    bool has_sparse_energy_current_operator,
    jams::MultiArray<double, 2>& dev_spin_derivative,
    jams::MultiArray<double, 2>& dev_energy_current_rx,
    jams::MultiArray<double, 2>& dev_energy_current_ry,
    jams::MultiArray<double, 2>& dev_energy_current_rz,
    jams::MultiArray<double, 1>& dev_energy_current_dot
);

jams::Vec<double, 3> execute_cuda_thermal_current_field_reduction(
    CudaStream& stream,
    double current_density_prefactor,
    const jams::MultiArray<double, 2>& dev_spin_derivative,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_rx,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_ry,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_rz,
    jams::MultiArray<double, 1>& dev_energy_current_dot);

class CudaDipoleFFTHamiltonian;
class Solver;

class CudaThermalCurrentMonitor : public Monitor {
public:
    CudaThermalCurrentMonitor(const libconfig::Setting &settings);
    ~CudaThermalCurrentMonitor();

    void update(Solver& solver);
    void post_process() override {};

private:
    CudaStream stream;

    jams::output::TsvWriter tsv_;

    double current_density_prefactor_ = 0.0;
    bool has_sparse_energy_current_operator_ = false;

    jams::SparseMatrix<double> energy_current_operator_rx_;
    jams::SparseMatrix<double> energy_current_operator_ry_;
    jams::SparseMatrix<double> energy_current_operator_rz_;
    std::vector<CudaDipoleFFTHamiltonian*> dipole_fft_energy_current_providers_;

    jams::MultiArray<double, 2> spin_derivative_;
    jams::MultiArray<double, 2> energy_current_rx_;
    jams::MultiArray<double, 2> energy_current_ry_;
    jams::MultiArray<double, 2> energy_current_rz_;
    jams::MultiArray<double, 1> energy_current_dot_;
    jams::MultiArray<jams::Real, 2> dipole_fft_energy_current_rx_;
    jams::MultiArray<jams::Real, 2> dipole_fft_energy_current_ry_;
    jams::MultiArray<jams::Real, 2> dipole_fft_energy_current_rz_;
};

#endif //JAMS_CUDA_THERMAL_CURRENT_H
