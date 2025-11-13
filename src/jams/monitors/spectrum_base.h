//
// Created by Joseph Barker on 2019-08-01.
//

#ifndef JAMS_SPECTRUM_BASE_H
#define JAMS_SPECTRUM_BASE_H

#include <jams/containers/multiarray.h>
#include <jams/core/globals.h>
#include <jams/core/monitor.h>
#include <jams/core/solver.h>
#include <jams/interface/fft.h>
#include <jams/helpers/defaults.h>

namespace jams {
    struct HKLIndex {
        Vec<double, 3> hkl; // reciprocal lattice point in fractional units
        Vec<double, 3> xyz; // reciprocal lattice point in cartesian units
        FFTWHermitianIndex<3> index;
    };

    inline bool operator==(const HKLIndex &a, const HKLIndex &b) {
      return approximately_equal(a.hkl, b.hkl, jams::defaults::lattice_tolerance);
    }
}

class SpectrumBaseMonitor : public Monitor {
public:
    using Complex = std::complex<double>;
    using CmplxVecField = jams::MultiArray<Vec3cx, 3>;

    explicit SpectrumBaseMonitor(const libconfig::Setting &settings);
    ~SpectrumBaseMonitor() override = default;

    void post_process() override = 0;
    void update(Solver& solver) override = 0;

    inline int num_motif_atoms() const {
      return num_motif_atoms_;
    }

    inline int num_kpoints() const {
      return kspace_paths_.size();
    }

    inline int num_time_samples() const {
      return periodogram_props_.length;
    };

    inline double sample_time_interval() const {
      return output_step_freq_ * globals::solver->time_step();
    }

    inline double num_periodogram_iterations() const {
      return total_periods_;
    }

    inline double frequency_resolution_thz() const {
      return (1.0 / (num_time_samples() * sample_time_interval()));
    }
    inline double max_frequency_thz() const {
      return (1.0 / (2.0 * sample_time_interval()));
    }

    void print_info() const;


protected:

    void configure_kspace_paths(libconfig::Setting& settings);
    void configure_periodogram(libconfig::Setting& settings);

    bool do_periodogram_update() const;
    void store_periodogram_data(const jams::MultiArray<double, 2> &data);

    CmplxVecField compute_periodogram_spectrum(CmplxVecField &timeseries);
    static void shift_periodogram_timeseries(CmplxVecField &timeseries, int overlap);

    static CmplxVecField fft_timeseries_to_frequency(CmplxVecField spectrum);

    static std::vector<jams::HKLIndex> generate_hkl_kspace_path(
        const std::vector<Vec3> &hkl_nodes, const Vec3i &kspace_size);

    void store_kspace_data_on_path(const jams::MultiArray<Vec3cx,4> &kspace_data, const std::vector<jams::HKLIndex> &kspace_path);

    std::vector<jams::HKLIndex> kspace_paths_;
    std::vector<int>            kspace_continuous_path_ranges_;

    jams::MultiArray<Vec3cx,4>  kspace_data_;
    CmplxVecField  kspace_data_timeseries_;



private:
    jams::PeriodogramProps periodogram_props_;
    int periodogram_index_ = 0;
    int total_periods_ = 0;
    int num_motif_atoms_ = 0;
};

#endif //JAMS_SPECTRUM_BASE_H
