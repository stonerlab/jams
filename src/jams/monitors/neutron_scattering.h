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

struct HKLIndex {
    Vec<double,3> hkl;        /**< reciprocal lattice point in fractional units */
    Vec<double,3> xyz;        /**< reciprocal lattice point in cartesian units */
    Vec<int,3>    index;      /**< array lookup index */
    bool          conjugate;  /**< does the value need conjugating on lookup, e.g. for r2c symmetry */
};

struct WelchParameters {
    int segment_size = {1024};
    int overlap      = {512};
};

inline bool operator==(const HKLIndex& a, const HKLIndex& b) {
  return approximately_equal(a.hkl, b.hkl);
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
    ~NeutronScatteringMonitor() override;

    void post_process() override {};
    void update(Solver *solver) override;
    bool is_converged() override { return false; }

private:

    HKLIndex fftw_remap_index_real_to_complex(Vec<int,3> k, const Vec<int,3> &N);

    std::vector<HKLIndex> generate_hkl_reciprocal_space_path(
        const std::vector<Vec3> &hkl_nodes, const Vec3i &reciprocal_space_size);

    void output_cross_section();
    jams::MultiArray<Complex, 2> compute_unpolarized_cross_section(const jams::MultiArray<Vec<Complex,3>, 3>& spectrum);
    jams::MultiArray<Complex, 3> compute_polarized_cross_sections(const jams::MultiArray<Vec<Complex,3>, 3>& spectrum, const std::vector<Vec3>& polarizations);

    fftw_plan fft_plan_transform_to_reciprocal_space(
        double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & num_sites);

    jams::MultiArray<Vec<Complex,3>,3> periodogram();

    fftw_plan fft_plan_to_qspace_ = nullptr;

    jams::MultiArray<Vec<Complex,3>, 4> sq_;
    jams::MultiArray<Vec<Complex,3>, 3> sqw_;
    jams::MultiArray<Complex, 2> total_unpolarized_cross_section_;
    jams::MultiArray<Complex,3> total_polarized_cross_sections_;

    jams::MultiArray<double, 2> form_factors_;
    std::vector<HKLIndex> paths_;
    std::vector<int> continuous_path_ranges_;
    int num_t_samples_ = 0;
    double t_sample_ = 0.0;
    std::vector<Vec3> polarizations_;
    double freq_delta_;
    int periodogram_index_counter_;
    int periodogram_counter_;
    WelchParameters welch_params_;
};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

