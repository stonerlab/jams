//
// Created by Joe Barker on 2017/10/04.
//

#ifndef JAMS_CUDA_SPIN_CURRENT_H
#define JAMS_CUDA_SPIN_CURRENT_H

#include <fstream>
#include <libconfig.h++>

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_sparse_interaction_matrix.h"
#include "jams/containers/sparsematrix.h"
#include "jams/core/monitor.h"
#include "jams/core/types.h"
#include "jblib/containers/cuda_array.h"

class Solver;

Mat3 execute_cuda_spin_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const double *dev_Jrij,
        const int *dev_col_pointers,
        const int *dev_col_indicies,
        double *dev_spin_current_rx_x,
        double *dev_spin_current_rx_y,
        double *dev_spin_current_rx_z,
        double *dev_spin_current_ry_x,
        double *dev_spin_current_ry_y,
        double *dev_spin_current_ry_z,
        double *dev_spin_current_rz_x,
        double *dev_spin_current_rz_y,
        double *dev_spin_current_rz_z
                                     );
class CudaSpinCurrentMonitor : public Monitor {
public:
    CudaSpinCurrentMonitor(const libconfig::Setting &settings);
    ~CudaSpinCurrentMonitor();

    void update(Solver * solver);
    bool is_converged();

private:
    void write_spin_current_h5_file(const std::string &h5_file_name);
    void open_new_xdmf_file(const std::string &xdmf_file_name);
    void update_xdmf_file(const std::string &h5_file_name);

    CudaStream stream;

    bool do_h5_output;
    unsigned h5_output_steps;
    FILE*        xdmf_file_ = nullptr;

    std::ofstream outfile;

    CudaSparseMatrixCSR<double> dev_csr_matrix_;

    jblib::CudaArray<double, 1> dev_spin_current_rx_x;
    jblib::CudaArray<double, 1> dev_spin_current_rx_y;
    jblib::CudaArray<double, 1> dev_spin_current_rx_z;

    jblib::CudaArray<double, 1> dev_spin_current_ry_x;
    jblib::CudaArray<double, 1> dev_spin_current_ry_y;
    jblib::CudaArray<double, 1> dev_spin_current_ry_z;

    jblib::CudaArray<double, 1> dev_spin_current_rz_x;
    jblib::CudaArray<double, 1> dev_spin_current_rz_y;
    jblib::CudaArray<double, 1> dev_spin_current_rz_z;

};

#endif //JAMS_CUDA_SPIN_CURRENT_H
