//
// Created by Joseph Barker on 2018-11-22.
//

#ifndef JAMS_CUDA_NEUTRON_SCATTERING_NOLATTICE_H
#define JAMS_CUDA_NEUTRON_SCATTERING_NOLATTICE_H

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include "jams/monitors/spectrum_general.h"
#include "jams/interface/fft.h"

class CudaSpectrumGeneralMonitor : public Monitor {
public:
    explicit CudaSpectrumGeneralMonitor(const libconfig::Setting &settings);
    ~CudaSpectrumGeneralMonitor() override = default;

    void post_process() override {};
    void update(Solver *solver) override;
    bool is_converged() override { return false; }
private:

    void configure_kspace_vectors(const libconfig::Setting& settings);

    void store_kspace_data_on_path();

    jams::MultiArray<Vec3cx,2> periodogram();
    void shift_periodogram_overlap();

    void output_neutron_cross_section();

    jams::MultiArray<Complex, 2> calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx,2>& spectrum);
    jams::MultiArray<Complex, 3> calculate_polarized_cross_sections(const jams::MultiArray<Vec3cx,2>& spectrum, const std::vector<Vec3>& polarizations);

    jams::MultiArray<Vec3, 1> rspace_displacement_;
    jams::MultiArray<double, 1> rspace_window_;
    jams::MultiArray<Vec3, 1> kspace_path_;
    jams::MultiArray<Vec3cx,2>  kspace_spins_timeseries_;

    jams::MultiArray<double, 2> neutron_form_factors_;
    std::vector<Vec3>           neutron_polarizations_;
    jams::MultiArray<Complex,2> total_unpolarized_neutron_cross_section_;
    jams::MultiArray<Complex,3> total_polarized_neutron_cross_sections_;

    jams::PeriodogramProps periodogram_props_;
    int periodogram_index_ = 0;
    int total_periods_ = 0;

    double kmax_ = 60.0;
    int num_k_ = 60;
    Vec3 kvector_ = {0, 0, 1};

    void configure_polarizations(libconfig::Setting &setting);

    void configure_periodogram(libconfig::Setting &setting);
};

#endif //JAMS_CUDA_NEUTRON_SCATTERING_NOLATTICE_H
