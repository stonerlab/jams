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

  /// @brief Enum defining which channel mapping scheme to use to map
  /// Sx,Sy,Sz to different channels for the spectrum.
  ///
  /// @details
  /// `Cartesian` leaves the channels as (Sx, Sy, Sz)
  /// `RaiseLower` maps to raising and lowering like operators (S+, S-, Sz) with
  /// S+ = (Sx + i Sy)/sqrt(2), S- = (Sx - i Sy)/sqrt(2) and scales the data by the spin size.
  enum class ChannelMapping
  {
    Cartesian, ///< (Sx, Sy, Sz)
    RaiseLower      ///< (S+, S-, Sz)
  };

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

    inline int periodogram_index() const {
      return periodogram_index_;
    }

    inline int periodogram_overlap() const {
      return periodogram_props_.overlap;
    }

    inline int num_periodogram_samples() const {
      return periodogram_props_.length;
    };

    inline double sample_time_interval() const {
      return output_step_freq_ * globals::solver->time_step();
    }

    inline double num_periodogram_periods() const {
      return total_periods_;
    }

    inline double frequency_resolution_thz() const {
      return (1.0 / (num_periodogram_samples() * sample_time_interval()));
    }
    inline double max_frequency_thz() const {
      return (1.0 / (2.0 * sample_time_interval()));
    }

    inline ChannelMapping get_channel_mapping() const { return channel_mapping_; }
    inline void set_channel_mapping(const ChannelMapping channel_mapping) { channel_mapping_ = channel_mapping; }

    void print_info() const;


protected:

    void insert_full_kspace(Vec3i kspace_size);
    void insert_continuous_kpath(libconfig::Setting& settings);
    void configure_kspace_paths(libconfig::Setting& settings);
    void configure_periodogram(libconfig::Setting& settings);

    bool do_periodogram_update() const;

    /// @brief Fourier transform S(r) -> S(k) and store in the timeseries S(k,t)
    ///
    /// @param [in] data Spin data S(r)
    void fourier_transform_to_kspace_and_store(const jams::MultiArray<double, 2> &data);

    /// @brief Applies the channel mapping to S(k,t) timeseries data.
    ///
    /// @details The channel mapping allows Sx, Sy, Sz components to be recombined. The most obvious use for this
    /// is to construct S+,S-,Sz.
    void apply_raise_lower_mapping(CmplxVecField& sk_timeseries, const jams::MultiArray<Mat3, 1>& rotations);


    CmplxVecField compute_periodogram_spectrum(CmplxVecField &timeseries);
    CmplxVecField compute_periodogram_rotated_spectrum(CmplxVecField &timeseries, const jams::MultiArray<Mat3, 1>& rotations);

    static void shift_periodogram_timeseries(CmplxVecField &timeseries, int overlap);


    static CmplxVecField fft_timeseries_to_frequency(CmplxVecField spectrum);

    static std::vector<jams::HKLIndex> generate_hkl_kspace_path(
        const std::vector<Vec3> &hkl_nodes, const Vec3i &kspace_size);

    void store_kspace_data_on_path(const jams::MultiArray<Vec3cx,4> &kspace_data, const std::vector<jams::HKLIndex> &kspace_path);

    bool do_full_kspace_ = false;

    std::vector<jams::HKLIndex> kspace_paths_;
    std::vector<int>            kspace_continuous_path_ranges_;

    /// @brief S(k) per basis site from the fourier transform of a single time
    /// @details Layout: kspace_data_(kx, ky, kz, basis_index)
    jams::MultiArray<Vec3cx,4> kspace_data_;

    /// @brief S(k, t) time series where only k along kpath are stored
    /// @details Layout: kspace_data_timeseries_(basis_index, periodogram_index, kpath_index)
    CmplxVecField kspace_data_timeseries_;

private:

  /// @brief 3x3 complex matrix that defines how the Sx, Sy, Sz components are combined before
  /// the Fourier transform. Defaults to identity matrix;
  ChannelMapping channel_mapping_ = ChannelMapping::RaiseLower;

  jams::PeriodogramProps periodogram_props_ {0, 0};
  int periodogram_index_ = 0;
  int total_periods_ = 0;
  int num_motif_atoms_ = 0;

  /// @brief Output memory for the spectrum
  jams::MultiArray<Vec3cx,3> skw_spectrum_;

};

#endif //JAMS_SPECTRUM_BASE_H
