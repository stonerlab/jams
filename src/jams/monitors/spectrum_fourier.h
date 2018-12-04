// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_STRUCTUREFACTOR_H
#define JAMS_MONITOR_STRUCTUREFACTOR_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

class Solver;

class SpectrumFourierMonitor : public Monitor {
 public:
  SpectrumFourierMonitor(const libconfig::Setting &settings);
  ~SpectrumFourierMonitor() override;

    void update(Solver *solver) override;

    bool is_converged() override { return false; }

private:
    void fft_space();

    void fft_time();

    void store_bz_path_data();

    fftw_plan fft_plan_s_rspace_to_kspace = nullptr;

    jblib::Array<fftw_complex, 5> s_kspace;

    bool output_sublattice_enabled_ = false;
    std::vector<Mat3> spin_transformations;
    jblib::Array<fftw_complex, 2> transformed_spins;
    jblib::Array<fftw_complex, 3> sqw_x;
    jblib::Array<fftw_complex, 3> sqw_y;
    jblib::Array<fftw_complex, 3> sqw_z;
    std::vector<Vec3i> b_uvw_nodes;
    std::vector<Vec3i> b_uvw_points;
    std::vector<int> bz_points_path_count;
    std::vector<double> bz_lengths;
    double freq_delta;
    int time_point_counter_;
};

#endif  // JAMS_MONITOR_STRUCTUREFACTOR_H
