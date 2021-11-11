// cuda_neutron_scattering_no_lattice.h                                -*-C++-*-
#ifndef INCLUDED_JAMS_MONITOR_CUDA_NEUTRON_SCATTERING_NO_LATTICE
#define INCLUDED_JAMS_MONITOR_CUDA_NEUTRON_SCATTERING_NO_LATTICE

#include "jams/monitors/spectrum_general.h"
#include "jams/interface/fft.h"
#include <jams/lattice/interaction_neartree.h>
#include <jams/core/monitor.h>

class CudaNeutronScatteringNoLatticeMonitor : public Monitor {
public:
    explicit CudaNeutronScatteringNoLatticeMonitor(const libconfig::Setting &settings);
    ~CudaNeutronScatteringNoLatticeMonitor() override = default;

    void post_process() override {};
    void update(Solver *solver) override;
    bool is_converged() override { return false; }
private:

    void configure_kspace_vectors(const libconfig::Setting& settings);
//    void configure_polarizations(libconfig::Setting &setting);
    void configure_periodogram(libconfig::Setting &setting);
//    void configure_form_factors(libconfig::Setting &settings);

    void store_spin_data();

    void shift_periodogram_overlap();

    void output_fixed_spectrum();

    bool do_rspace_windowing_ = true;
    jams::MultiArray<Vec3, 1>   kspace_path_;
    jams::MultiArray<double,3> spin_timeseries_;
    jams::MultiArray<std::complex<double>,3> spin_frequencies_;

    jams::MultiArray<double, 2> neutron_form_factors_;
    std::vector<Vec3>           neutron_polarizations_;
    jams::MultiArray<Complex,2> total_unpolarized_neutron_cross_section_;
    jams::MultiArray<Complex,3> total_polarized_neutron_cross_sections_;

    jams::PeriodogramProps periodogram_props_;
    int periodogram_index_ = 0;
    int total_periods_ = 0;

    double kmax_ = 50.0;
    int num_k_ = 100;
    Vec3 kvector_ = {0.0, 0.0, 1.0};

};

#endif
// ----------------------------- END-OF-FILE ----------------------------------