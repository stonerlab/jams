//
// Created by Joe Barker on 2018/04/20.
//

#ifndef JAMS_CUDA_THERMAL_CURRENT_H
#define JAMS_CUDA_THERMAL_CURRENT_H
#include <fstream>
#include "jams/core/monitor.h"
#include "jams/cuda/cuda_stream.h"
#include "jams/core/interactions.h"
#include "jams/core/types.h"
#include "jams/containers/interaction_matrix.h"

Vec3 execute_cuda_thermal_current_kernel(
    CudaStream &stream,
    const jams::MultiArray<double, 2>& spins,
    const jams::InteractionMatrix<Vec3, double>& interaction_matrix,
    jams::MultiArray<double, 1>& dev_thermal_current_rx,
    jams::MultiArray<double, 1>& dev_thermal_current_ry,
    jams::MultiArray<double, 1>& dev_thermal_current_rz
);

class Solver;

class CudaThermalCurrentMonitor : public Monitor {
public:
    CudaThermalCurrentMonitor(const libconfig::Setting &settings);
    ~CudaThermalCurrentMonitor();

    void update(Solver& solver);
    void post_process() override {};

    inline std::string name() const {return "cuda-thermal-current";}


private:
    CudaStream stream;

    using ThreeSpinList = jams::InteractionList<Vec3, 3>;

    ThreeSpinList generate_three_spin_from_two_spin_interactions(const jams::InteractionList<Mat3, 2>& nbr_list);

    std::ofstream outfile;

    jams::InteractionMatrix<Vec3, double> interaction_matrix_;

    jams::MultiArray<double, 1> thermal_current_rx_;
    jams::MultiArray<double, 1> thermal_current_ry_;
    jams::MultiArray<double, 1> thermal_current_rz_;
};

#endif //JAMS_CUDA_THERMAL_CURRENT_H
