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
#include <complex>

namespace jams {

    /// @brief Struct to store information about kpoints
    struct HKLIndex {
        Vec<double, 3> hkl; ///< reciprocal lattice point in fractional units
        Vec<double, 3> xyz; ///< reciprocal lattice point in cartesian units
        FFTWHermitianIndex<3> index; ///< FFTW 3D array index and conjugation
    };

    inline bool operator==(const HKLIndex &a, const HKLIndex &b) {
      return jams::approximately_equal(a.hkl, b.hkl, jams::defaults::lattice_tolerance);
    }
}

class SpectrumBaseMonitor : public Monitor {
public:
  /// @brief Defines how k-space is sampled for spectral calculations.
  ///
  /// Controls whether the monitor reads a user-specified k-space path from the
  /// configuration file or automatically generates a full Brillouin-zone sampling.
  enum class KSamplingMode
  {
    /// Use the k-space path specified in the input settings (e.g. hkl_path).
    /// Allows arbitrary user-defined paths or lists of paths.
    FromSettings,

    /// Ignore any path in the settings and instead generate a full k-space grid
    /// covering the entire Brillouin zone of the simulation supercell.
    FullGrid
  };

/// @brief Defines the mapping from Cartesian spin components to output spectral channels.
///
/// This structure specifies how spin components are transformed and combined before
/// Fourier transforming the time series. It allows construction of arbitrary channel
/// combinations (e.g. Cartesian components, circular components S⁺/S⁻, or custom
/// linear combinations).
///
/// The mapping is applied to the spin vector (optionally rotated into the local
/// sublattice frame and scaled by spin length) using a linear transformation:
///
/// \f[
/// C_i = \sum_{j \in \{x,y,z\}} M_{ij}\,S_j
/// \f]
///
/// where \f$M\f$ is the complex coefficient matrix given by @ref coeffs and
/// \f$C_i\f$ are the resulting output channels.
///
/// Typical uses include:
/// - Cartesian mapping: identity matrix → (Sx, Sy, Sz)
/// - Circular mapping: rows for S⁺, S⁻, Sz
/// - Single-channel mapping: e.g. S⁺ only for magnon density
struct ChannelTransform
{
  /// Number of output channels produced by the mapping.
  ///
  /// Must be in the range [1, 3]. Only the first @ref output_channels rows of
  /// @ref coeffs are used.
  int output_channels = 3;

  /// Complex coefficient matrix defining the linear mapping from (Sx,Sy,Sz)
  /// to output channels.
  ///
  /// Each row corresponds to one output channel. For example:
  /// - Identity → Cartesian channels
  /// - Circular combinations → S⁺, S⁻
  Mat3cx weights = kIdentityMat3cx;

  /// If true, spins are first rotated into a local basis reference frame
  /// before applying the channel mapping.
  ///
  /// The local z-axis is taken along the time-averaged sublattice magnetisation
  /// direction within the current periodogram window. This is typically required
  /// for non-collinear systems or for constructing transverse spin components.
  bool use_local_frame = false;

  /// If true, spin components are scaled by the physical spin length before
  /// applying the channel mapping.
  ///
  /// This converts unit spins into physical spin units using
  /// \f$S = \mu/g\f$, where \f$\mu\f$ is the magnetic moment associated with
  /// the site. Required when constructing physically meaningful quantities
  /// such as magnon occupation.
  bool scale_to_physical_spin = false;
};

  using CmplxMappedField = jams::MultiArray<jams::ComplexHi, 4>;     // (site, time, k, channel)
  using CmplxStored = std::complex<float>;
  using CmplxStoredField = jams::MultiArray<CmplxStored, 4>;          // (site, time, k, channel), compact storage
  using CmplxMappedSpectrum = jams::MultiArray<jams::ComplexHi, 4>;  // (site, freq, k, channel)
  using CmplxMappedSlice = jams::MultiArray<jams::ComplexHi, 3>;     // (site, freq, channel)

  explicit SpectrumBaseMonitor(
    const libconfig::Setting &settings, KSamplingMode k_sampling_mode = KSamplingMode::FromSettings);

  ~SpectrumBaseMonitor() override;

  void post_process() override = 0;
  void update(Solver& solver) override = 0;

  static ChannelTransform cartesian_channel_map();
  static ChannelTransform raise_lower_channel_map();


  /// @brief The number of atoms in the unit cell basis
  int num_basis_atoms() const { return num_basis_atoms_; }

  /// @brief The total number of k-points across all k-paths
  int num_k_points() const { return k_points_.size(); }

  /// @brief Returns true if we want to keep the negative frequencies
  bool keep_negative_frequencies() const { return keep_negative_frequencies_; }

  /// @brief Returns the number of sampled frequencies
  int num_frequencies() const
  {
    if (keep_negative_frequencies_)
    {
      return periodogram_length();
    }
    return periodogram_length() / 2 + 1;
  }

  /// @brief The sample index within the current periodogram
  int periodogram_sample_index() const { return periodogram_sample_index_; }

  /// @brief The number of overlapping samples of the periodogram
  int periodogram_overlap() const { return periodogram_props_.overlap; }

  /// @brief The number of samples in one periodogram period
  int periodogram_length() const { return periodogram_props_.length; }

  /// @brief The number of periodogram periods that have passed so far.
  ///
  /// @details This includes the current period, so for example, during the first periodogram this will return 1.
  int periodogram_window_count() const { return periodogram_window_count_; }

  /// @brief The real time between periodogram samples in picoseconds
  double sample_time_interval() const
  {
    return output_step_freq_ * globals::solver->time_step();
  }

  /// @brief The frequency resolution of the spectrum in THz
  double frequency_resolution_thz() const
  {
    return (1.0 / (periodogram_length() * sample_time_interval()));
  }

  /// @brief Maximum (Nyquist) frequency of the spectrum in THz
  double max_frequency_thz() const
  {
    return (1.0 / (2.0 * sample_time_interval()));
  }

  const ChannelTransform& channel_map() const { return channel_transform_; }
  void set_channel_map(const ChannelTransform& channel_map);
  int num_channels() const { return channel_transform_.output_channels; }

  void print_info() const;

protected:
  /// @brief Advance to the next periodogram window, preserving the configured overlap.
  void advance_periodogram_window();

  void append_full_k_grid(Vec3i kspace_size);
  void append_k_path_segment(libconfig::Setting& settings);
  void configure_k_list(libconfig::Setting& settings);
  void configure_periodogram(libconfig::Setting& settings);

  bool periodogram_window_complete() const;

  /// @brief Update the stored k-space time series from the current real-space spin state.
  ///
  /// Performs a spatial Fourier transform S(r) → S(k) and appends the result to the
  /// current periodogram window.
  void store_sk_snapshot(const jams::MultiArray<double,2>& spin_state);

  const CmplxMappedSpectrum& finalise_periodogram_spectrum();

  /// @brief Compute the frequency-domain spectrum S(k,ω) for a single k-point.
  ///
  /// Uses the stored k-space time series S(k,t) for the current periodogram window,
  /// applies windowing and mean removal, and performs the temporal Fourier transform.
  /// The returned slice contains the complex spectrum for all sites and channels
  /// at the specified k-point.
  ///
  /// @param kpoint_index Index of the k-point in the stored k-space path.
  /// @return Reference to internal buffer containing S(k,ω) for this k-point.
  const CmplxMappedSlice& compute_frequency_spectrum_at_k(int kpoint_index);

  static void make_hkl_path(
      const std::vector<Vec3> &hkl_nodes,
      const Vec3i &kspace_size,
      std::vector<jams::HKLIndex>& hkl_path);

  /// @brief Append one time sample of S(k) for all k-points on the configured path.
  ///
  /// Extracts S(k) values for each basis site and each k-point in @p kspace_path from
  /// the full k-space field @p kspace_data, applies the unit-cell basis phase factors,
  /// and performs Hermitian reconstruction (conjugation) where required by the r2c layout.
  /// The resulting values are stored into the k-space time series buffer at the current
  /// periodogram index.
  ///
  /// If dynamic channel mapping is enabled, Cartesian components (Sx,Sy,Sz) are stored
  /// for later rotation/mapping; otherwise the configured channel map is applied here.
  ///
  /// @param sk_sample Full k-space field for the current time step (FFT output).
  /// @param k_list List of k-points to extract (path or full grid) including r2c indices.
  void append_sk_sample_for_k_list(const jams::MultiArray<Vec3cx,4>& sk_sample,
                                 const std::vector<jams::HKLIndex>& k_list);

  jams::MultiArray<Mat3, 1> generate_sublattice_rotations_();

  std::vector<jams::HKLIndex> k_points_;
  std::vector<int>            k_segment_offsets_;

  /// @brief S(k) per basis site from the fourier transform of a single time
  /// @details Layout: sk_grid_(kx, ky, kz, basis_index)
  jams::MultiArray<Vec3cx,4> sk_grid_;

  /// @brief S(k, t) time series where only k along kpath are stored
  /// @details Layout: sk_time_series_(basis_index, periodogram_index, kpath_index, channel)
  jams::MultiArray<jams::ComplexHi, 2> basis_phase_factors_;
  CmplxStoredField sk_time_series_;

private:
  void store_sublattice_magnetisation_(const jams::MultiArray<double, 2> &spin_state);
  jams::MultiArray<Vec3, 1> compute_mean_basis_mag_directions_();

  jams::ComplexHi map_spin_component_(int basis_index,
                                      int channel_index,
                                      const Vec3cx& spin_xyz,
                                      const jams::MultiArray<Mat3, 1>* rotations) const;
  Vec3cx read_cartesian_spin_(int basis_index, int time_index, int k_index) const;
  bool needs_local_frame_mapping_() const;
  void resize_channel_storage_();

  /// @brief Generate the window function for a width of num_time_samples.
  ///
  /// @details FFT window is normalised to unit RMS power so that windowing does not change overall power.
  static void generate_normalised_window_(jams::MultiArray<double,1>& window, int num_time_samples);

  /// @brief Generate unit cell phase factors phi_a(k) = exp(-2 pi i r_a.k) where a is a position in the unit cell basis
  /// and k is a k point.
  static void generate_phase_factors_(
      jams::MultiArray<jams::ComplexHi, 2>& phase_factors,
      const std::vector<Vec3>& r_frac,
      const std::vector<jams::HKLIndex>& kpoints);

  bool keep_negative_frequencies_ = false;
  ChannelTransform channel_transform_ = cartesian_channel_map();

  jams::PeriodogramProps periodogram_props_ {2000, 1000};
  int periodogram_sample_index_ = 0;
  int periodogram_window_count_ = 1; // First period is period 1
  int num_basis_atoms_ = 0;

  /// @brief Sublattice magnetisation directions for each basis site at each time sample in the current periodogram
  /// @details Layout: mean_sublattice_directions_(basis_site, periodogram_index)
  jams::MultiArray<Vec3, 2> basis_mag_time_series_;

  int stored_channel_count_ = 0;

  /// @brief Output memory for the mapped spectrum
  CmplxMappedSpectrum skw_buffer_;

  /// @brief Buffer for FFT S(k,t...) -> S(k,w) for a single k index
  fftw_plan sk_time_fft_plan_ = nullptr;
  jams::MultiArray<double,1> periodogram_window_;
  CmplxMappedSlice frequency_scratch_;
};

#endif //JAMS_SPECTRUM_BASE_H
