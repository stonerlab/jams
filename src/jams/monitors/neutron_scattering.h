// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_NEUTRON_SCATTERING_H
#define JAMS_MONITOR_NEUTRON_SCATTERING_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jblib/containers/array.h"

class Solver;

struct HKLIndex {
    Vec<double,3> hkl;        /**< reciprocal lattice point in fractional units */
    Vec<double,3> xyz;        /**< reciprocal lattice point in cartesian units */
    Vec<int,3>    index;      /**< array lookup index */
    bool          conjugate;  /**< does the value need conjugating on lookup, e.g. for r2c symmetry */
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

    void update(Solver *solver) override;
    void post_process() override;
    bool is_converged() override { return false; }

private:

    HKLIndex fftw_remap_index_real_to_complex(Vec<int,3> k, const Vec<int,3> &N);

    std::vector<HKLIndex> generate_hkl_reciprocal_space_path(
        const std::vector<Vec3> &hkl_nodes, const Vec3i &reciprocal_space_size);

    jams::MultiArray<Complex, 2> unpolarized_partial_cross_section();

    fftw_plan fft_plan_transform_to_reciprocal_space(
        double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & num_sites);

    void fft_to_frequency();

    fftw_plan fft_plan_to_qspace_ = nullptr;

    jams::MultiArray<Complex, 5> sq_;
    jams::MultiArray<Complex, 4> sqw_;
    std::vector<HKLIndex> path_;
    double freq_delta_;
    int time_point_counter_;
};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

