//
// Created by Joe Barker on 2018/04/20.
//

#ifndef JAMS_CUDA_THERMAL_CURRENT_H
#define JAMS_CUDA_THERMAL_CURRENT_H
#include <fstream>
#include "jams/core/monitor.h"
#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_sparse_interaction_matrix.h"
#include "jblib/containers/cuda_array.h"
#include "jams/core/interactions.h"
#include "jams/core/types.h"

Vec3 execute_cuda_thermal_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const int *index_pointers,
        const int *index_data,
        const double *value_data,
        double *dev_thermal_current_rx,
        double *dev_thermal_current_ry,
        double *dev_thermal_current_rz
                                        );

class Solver;

class CudaThermalCurrentMonitor : public Monitor {
public:
    CudaThermalCurrentMonitor(const libconfig::Setting &settings);
    ~CudaThermalCurrentMonitor();

    void update(Solver * solver);
    bool is_converged();
    inline std::string name() const {return "cuda-thermal-current";}


private:
    CudaStream stream;

    using TriadList = std::vector<Triad<Vec3>>;

    TriadList generate_triads_from_neighbour_list(const InteractionList<Mat3>& nbr_list);
    void initialize_device_data(const TriadList& triads);

    std::ofstream outfile;

    CudaSparseMatrixCSR<double> dev_csr_matrix_;

    jblib::CudaArray<double, 1> dev_thermal_current_rx;
    jblib::CudaArray<double, 1> dev_thermal_current_ry;
    jblib::CudaArray<double, 1> dev_thermal_current_rz;
};

#endif //JAMS_CUDA_THERMAL_CURRENT_H
