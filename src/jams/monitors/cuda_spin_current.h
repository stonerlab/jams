//
// Created by Joe Barker on 2017/10/04.
//

#ifndef JAMS_CUDA_SPIN_CURRENT_H
#define JAMS_CUDA_SPIN_CURRENT_H

#include <jams/containers/sparse_matrix.h>
#include <jams/core/monitor.h>
#include <jams/core/types.h>
#include <jams/cuda/cuda_stream.h>

#include <fstream>

class Solver;

Vec3 execute_cuda_spin_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const jams::Real *dev_gyro,
        const jams::Real *dev_mus,
        const double *dev_Jrij,
        const int *dev_col_pointers,
        const int *dev_col_indicies,
        double *dev_spin_current_rx_z,
        double *dev_spin_current_ry_z,
        double *dev_spin_current_rz_z
                                     );


/// @class CudaSpinCurrentMonitor
///
/// Calculates the instantaneous total spin current in the system.
///
/// @details
/// This monitor calculates the total Sz spin current from the equation
///
/// \vec{J}_{s}^{(z)} = \frac{1}{2} \sum_i\frac{\gamma_i}{\mu_i} \left(\sum_j (\vec{r}_i-\vec{r}_j)J_{ij}(S_i^x S_j^y - S_i^y S_j^x) \right)
///
/// The output units are in m/s. To get to a real spin current they should be
/// multiplied by ℏ/2 and to convert to spin current density they should be
/// divided by the simulation volume.
///
/// Spin current density in 'electical units' of electical current density A/m^2
/// would be: (2e/ℏ) * (ℏ/2) * (1/V) * j_s = (e/V) j_s
///
/// @warning the results are derived strictly for a system with a Hamiltonian
/// \[
/// \mathscr{H} = -\frac{1}{2}\sum_{ij} J_{ij} \vec{S}_i \cdot \vec{S}_j - \sum_{i}\mu_i B_z\cdot S_{z,i}
/// \]
/// Systems which have a different Hamiltonian, especially where there is
/// non-colinearity or the magnetisation doesn't lie along z are not considered.
///
/// @setting `h5` (optional) true|false for outputting h5 data of bond currents
///               (default false)
///
/// @setting `h5_output_steps` (optional) when h5 output is enabled this is the
///                            number of timesteps between outputs
///                            (default is same as monitor output_steps)
///
/// @example
/// @code
/// monitors = (
///   {
///     module = "spin-current";
///     h5 = true;
///     h5_output_steps = 100;
///   }
/// );
/// @endcode

class CudaSpinCurrentMonitor : public Monitor {
public:
    CudaSpinCurrentMonitor(const libconfig::Setting &settings);
    ~CudaSpinCurrentMonitor();

    void update(Solver& solver);
    void post_process() override {};

private:
    void write_spin_current_h5_file(const std::string &h5_file_name, const int iteration, const double time);
    void open_new_xdmf_file(const std::string &xdmf_file_name);
    void update_xdmf_file(const std::string &h5_file_name, const double time);

    CudaStream stream;

    bool do_h5_output;
    unsigned h5_output_steps;
    FILE*        xdmf_file_ = nullptr;

    std::ofstream outfile;

    jams::SparseMatrix<Vec3> interaction_matrix_;

    jams::MultiArray<double, 1> spin_current_rx_z;
    jams::MultiArray<double, 1> spin_current_ry_z;
    jams::MultiArray<double, 1> spin_current_rz_z;

};

#endif //JAMS_CUDA_SPIN_CURRENT_H
