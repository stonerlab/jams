// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_NEUTRON_SCATTERING_H
#define JAMS_MONITOR_NEUTRON_SCATTERING_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

class Solver;

namespace jams {
    struct HKLIndex {
        Vec<double, 3> hkl; // reciprocal lattice point in fractional units
        Vec<double, 3> xyz; // reciprocal lattice point in cartesian units
        FFTWHermitianIndex<3> index;
    };

    inline bool operator==(const HKLIndex &a, const HKLIndex &b) {
      return approximately_equal(a.hkl, b.hkl);
    }

    struct PeriodogramProps {
        int length    = 1000;
        int overlap    = 500;
        double sample_time  = 1.0;
    };
}

/**
 * Monitor for calculating neutron scattering cross-sections
 *
 * Neutron scattering cross-sections contain several factors which we don't
 * want in a 'pure' spin wave spectrum. For example:
 *
 *  - the magnetic structure factor which causes forbidden reflections depending
 * on the Brillouin zone
 *  - the polarization factor where only components perpendicular to Q can be
 * measured
 *  - extended zone scheme so we are not limited to the first Brillouin zone /
 * reduced zone scheme
 *
 * This monitor includes all of these effects allowing a direct
 * comparison to neutron scattering measurments.
 *
 */
class NeutronScatteringMonitor : public Monitor {
 public:
    using Complex = std::complex<double>;

    explicit NeutronScatteringMonitor(const libconfig::Setting &settings);
    ~NeutronScatteringMonitor() override = default;

    void post_process() override {};
    void update(Solver *solver) override;
    bool is_converged() override { return false; }

private:

    void print_info();

    void configure_kspace_paths(libconfig::Setting& settings);
    void configure_form_factors(libconfig::Setting& settings);
    void configure_polarizations(libconfig::Setting& settings);
    void configure_periodogram(libconfig::Setting& settings);

    void store_kspace_data_on_path();

    void shift_periodogram_overlap();
    jams::MultiArray<Vec3cx,3> periodogram();
    void output_neutron_cross_section();

    std::vector<jams::HKLIndex> generate_hkl_kspace_path(
        const std::vector<Vec3> &hkl_nodes, const Vec3i &kspace_size);

    jams::MultiArray<Complex, 2> calculate_unpolarized_cross_section(const jams::MultiArray<Vec3cx, 3>& spectrum);
    jams::MultiArray<Complex, 3> calculate_polarized_cross_sections(const jams::MultiArray<Vec3cx, 3>& spectrum, const std::vector<Vec3>& polarizations);

    std::vector<jams::HKLIndex> kspace_paths_;
    std::vector<int>            kspace_continuous_path_ranges_;
    jams::MultiArray<Vec3cx,4>  kspace_spins_;
    jams::MultiArray<Vec3cx,3>  kspace_spins_timeseries_;

    jams::MultiArray<double, 2> neutron_form_factors_;
    std::vector<Vec3>           neutron_polarizations_;
    jams::MultiArray<Complex,2> total_unpolarized_neutron_cross_section_;
    jams::MultiArray<Complex,3> total_polarized_neutron_cross_sections_;

    jams::PeriodogramProps periodogram_props_;
    int periodogram_index_ = 0;
    int total_periods_ = 0;

};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

