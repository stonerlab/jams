//
// Created by Joseph Barker on 2018-11-22.
//

#ifndef JAMS_NEUTRON_SCATTERING_NO_LATTICE_H
#define JAMS_NEUTRON_SCATTERING_NO_LATTICE_H

#include <jams/core/monitor.h>
#include <jams/interface/fft.h>
#include <jams/lattice/interaction_neartree.h>
#include <jams/helpers/mixed_precision.h>

class NeutronScatteringNoLatticeMonitor : public Monitor {
public:
    explicit NeutronScatteringNoLatticeMonitor(const libconfig::Setting &settings);
    ~NeutronScatteringNoLatticeMonitor() override = default;

    void post_process() override {};
    void update(Solver& solver) override;
private:

    void configure_kspace_vectors(const libconfig::Setting& settings);
    void configure_polarizations(libconfig::Setting &setting);
    void configure_periodogram(libconfig::Setting &setting);
    void configure_form_factors(libconfig::Setting &settings);

    void store_kspace_data_on_path();
    void store_spin_data();

    jams::MultiArray<Vec3cx,1> calculate_static_structure_factor();
    jams::MultiArray<Vec3cx,2> periodogram();
    void shift_periodogram_overlap();
    void output_neutron_cross_section();
    void output_static_structure_factor();

    void output_fixed_spectrum();

    jams::MultiArray<jams::ComplexHi, 2> calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx,2>& spectrum);
    jams::MultiArray<jams::ComplexHi, 3> calculate_polarized_cross_sections(const jams::MultiArray<Vec3cx,2>& spectrum, const std::vector<Vec3>& polarizations);

    bool do_rspace_windowing_ = true;
    jams::MultiArray<Vec3, 1> rspace_displacement_;
    jams::MultiArray<Vec3, 1> kspace_path_;
    jams::MultiArray<Vec3cx,2>  kspace_spins_timeseries_;

    jams::MultiArray<double,3> spin_timeseries_;
    jams::MultiArray<std::complex<double>,3> spin_frequencies_;

    jams::MultiArray<double, 2> neutron_form_factors_;
    std::vector<Vec3>           neutron_polarizations_;
    jams::MultiArray<jams::ComplexHi,2> total_unpolarized_neutron_cross_section_;
    jams::MultiArray<jams::ComplexHi,3> total_polarized_neutron_cross_sections_;

    jams::PeriodogramProps periodogram_props_;
    int periodogram_index_ = 0;
    int total_periods_ = 0;

    double kmax_ = 50.0;
    int num_k_ = 100;
    Vec3 kvector_ = {0.0, 0.0, 1.0};

    jams::InteractionNearTree neartree_;


};

#endif //JAMS_NEUTRON_SCATTERING_NO_LATTICE_H
