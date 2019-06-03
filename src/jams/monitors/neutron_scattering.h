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
};

inline bool operator==(const Qpoint& a, const Qpoint& b) {
  return approximately_equal(a.hkl, b.hkl);
}


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

    jams::MultiArray<Complex, 2> transformed_spins_;
    jams::MultiArray<Complex, 3> sqw_x_;
    jams::MultiArray<Complex, 3> sqw_y_;
    jams::MultiArray<Complex, 3> sqw_z_;
    std::vector<Vec3> brillouin_zone_nodes_;
    std::vector<Vec3i> brillouin_zone_mapping_;
    std::vector<Vec3> hkl_indicies_;
    std::vector<Vec3> q_vectors_;
    double freq_delta_;
    int time_point_counter_;
};

#endif  // JAMS_MONITOR_NEUTRON_SCATTERING_H

