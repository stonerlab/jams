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

#include <array>

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
  struct ChannelMap
  {
    int output_channels = 3;
    std::array<std::array<jams::ComplexHi, 3>, 3> coeffs {{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}
    }};
    bool rotate_to_sublattice_frame = false;
    bool scale_by_spin_length = false;
  };

  using CmplxMappedField = jams::MultiArray<jams::ComplexHi, 4>;     // (site, time, k, channel)
  using CmplxMappedSpectrum = jams::MultiArray<jams::ComplexHi, 4>;  // (site, freq, k, channel)
  using CmplxMappedSlice = jams::MultiArray<jams::ComplexHi, 3>;     // (site, freq, channel)

  explicit SpectrumBaseMonitor(const libconfig::Setting &settings);
  ~SpectrumBaseMonitor() override;

  void post_process() override = 0;
  void update(Solver& solver) override = 0;

  static ChannelMap cartesian_channel_map();
  static ChannelMap raise_lower_channel_map();

  int num_motif_atoms() const {
    return num_motif_atoms_;
  }

  int num_kpoints() const {
    return kspace_paths_.size();
  }

  bool keep_negative_frequencies() const
  {
    return keep_negative_frequencies_;
  }

  int num_frequencies() const
  {
    if (keep_negative_frequencies_)
    {
      return num_periodogram_samples();
    }
    return num_periodogram_samples() / 2 + 1;
  }

  int periodogram_index() const {
    return periodogram_index_;
  }

  int periodogram_overlap() const {
    return periodogram_props_.overlap;
  }

  int num_periodogram_samples() const {
    return periodogram_props_.length;
  }

  double sample_time_interval() const {
    return output_step_freq_ * globals::solver->time_step();
  }

  double num_periodogram_periods() const {
    return total_periods_;
  }

  double frequency_resolution_thz() const {
    return (1.0 / (num_periodogram_samples() * sample_time_interval()));
  }

  double max_frequency_thz() const {
    return (1.0 / (2.0 * sample_time_interval()));
  }

  const ChannelMap& channel_map() const { return channel_map_; }
  void set_channel_map(const ChannelMap& channel_map);
  int num_channels() const { return channel_map_.output_channels; }

  void print_info() const;

protected:
  /// @brief Resets the periodogram for a new period, shifting data by the overlap
  void shift_periodogram();

  void insert_full_kspace(Vec3i kspace_size);
  void insert_continuous_kpath(libconfig::Setting& settings);
  void configure_kspace_paths(libconfig::Setting& settings);
  void configure_periodogram(libconfig::Setting& settings);

  bool do_periodogram_update() const;

  /// @brief Fourier transform S(r) -> S(k) and store in the timeseries S(k,t)
  ///
  /// @param [in] data Spin data S(r)
  void fourier_transform_to_kspace_and_store(const jams::MultiArray<double, 2> &data);

  CmplxMappedSpectrum compute_periodogram_spectrum();
  CmplxMappedSpectrum compute_periodogram_rotated_spectrum();

  /// @brief Fourier transform the S(k,t) timeseries to S(k,w) at a single k-point
  CmplxMappedSlice& fft_sk_timeseries_to_skw(int kpoint_index);

  static std::vector<jams::HKLIndex> generate_hkl_kspace_path(
      const std::vector<Vec3> &hkl_nodes, const Vec3i &kspace_size);

  void store_kspace_data_on_path(const jams::MultiArray<Vec3cx,4> &kspace_data,
                                 const std::vector<jams::HKLIndex> &kspace_path);

  jams::MultiArray<Mat3, 1> generate_sublattice_rotations_();

  std::vector<jams::HKLIndex> kspace_paths_;
  std::vector<int>            kspace_continuous_path_ranges_;

  /// @brief S(k) per basis site from the fourier transform of a single time
  /// @details Layout: kspace_data_(kx, ky, kz, basis_index)
  jams::MultiArray<Vec3cx,4> kspace_data_;

  /// @brief S(k, t) time series where only k along kpath are stored
  /// @details Layout: sk_timeseries_(basis_index, periodogram_index, kpath_index, channel)
  jams::MultiArray<jams::ComplexHi, 2> sk_phase_factors_;
  CmplxMappedField sk_timeseries_;

private:
  void shift_sk_timeseries_();

  void store_sublattice_magnetisation_(const jams::MultiArray<double, 2> &spin_state);
  void shift_sublattice_magnetisation_timeseries_();
  jams::MultiArray<Vec3, 1> generate_sublattice_magnetisation_directions_();

  jams::ComplexHi map_spin_component_(int basis_index,
                                      int channel_index,
                                      const Vec3cx& spin_xyz,
                                      const jams::MultiArray<Mat3, 1>* rotations) const;
  bool requires_dynamic_channel_mapping_() const;
  void resize_channel_storage_();

  /// @brief Generate the window function for a width of num_time_samples.
  ///
  /// @details FFT window is normalised to unit RMS power so that windowing does not change overall power.
  static jams::MultiArray<double,1> generate_normalised_window_(int num_time_samples);

  /// @brief Generate unit cell phase factors phi_a(k) = exp(-2 pi i r_a.k) where a is a position in the unit cell basis
  /// and k is a k point.
  static jams::MultiArray<jams::ComplexHi, 2> generate_phase_factors_(
      const std::vector<Vec3>& r_frac,
      const std::vector<jams::HKLIndex>& kpoints);

  bool keep_negative_frequencies_ = false;
  ChannelMap channel_map_ = cartesian_channel_map();

  jams::PeriodogramProps periodogram_props_ {0, 0};
  int periodogram_index_ = 0;
  int total_periods_ = 0;
  int num_motif_atoms_ = 0;

  /// @brief Sublattice magnetisation directions for each basis site at each time sample in the current periodogram
  /// @details Layout: mean_sublattice_directions_(basis_site, periodogram_index)
  jams::MultiArray<Vec3, 2> sublattice_magnetisation_;

  /// @brief Cartesian S(k,t) timeseries when channel mapping depends on per-period local-frame rotations.
  jams::MultiArray<Vec3cx,3> sk_cartesian_timeseries_;

  /// @brief Output memory for the mapped spectrum
  CmplxMappedSpectrum skw_spectrum_;

  /// @brief Buffer for FFT S(k,t...) -> S(k,w) for a single k index
  fftw_plan sw_spectrum_fft_plan_ = nullptr;
  jams::MultiArray<double,1> sw_window_;
  CmplxMappedSlice sw_spectrum_buffer_;
};

#endif //JAMS_SPECTRUM_BASE_H
