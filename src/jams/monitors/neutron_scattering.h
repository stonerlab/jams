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


struct Qpoint {
    Vec3 hkl;
    Vec3 q;
    Vec3i index;
    bool hermitian;
};

inline bool operator==(const Qpoint& a, const Qpoint& b) {
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

    jams::MultiArray<std::complex<double>,2> fft_time_to_frequency(unsigned site, const jams::MultiArray<Complex, 3>& s_time);

    fftw_plan fft_plan_s_to_reciprocal_space_ = nullptr;

    jams::MultiArray<Complex, 5> s_reciprocal_space_;

    jams::MultiArray<Complex, 3> sqw_x_;
    jams::MultiArray<Complex, 3> sqw_y_;
    jams::MultiArray<Complex, 3> sqw_z_;
    std::vector<Vec3> brillouin_zone_nodes_;
    std::vector<Vec3i> brillouin_zone_mapping_;
    std::vector<Vec3> hkl_indicies_;
    std::vector<Vec3> q_vectors_;
    std::vector<bool> conjugation_;
    double freq_delta_;
    int time_point_counter_;
};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

