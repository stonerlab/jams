//
// Created by Joseph Barker on 2019-08-01.
//

#include "jams/monitors/spectrum_base.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/interface/config.h"
#include "jams/interface/fft.h"
#include "jams/interface/lapack_tridiagonal.h"
#include "jams/monitors/kpoint_path_builder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <stdexcept>
#include <vector>

#ifdef HAS_OMP
#include <omp.h>
#endif

namespace {

constexpr std::size_t kAutoFileBackedSkTimeSeriesThresholdBytes =
    12ull * 1024ull * 1024ull * 1024ull;

int default_fftw_thread_count()
{
#ifdef HAS_OMP
  return std::max(1, omp_get_max_threads());
#else
  const auto hw_threads = std::thread::hardware_concurrency();
  return hw_threads > 0 ? static_cast<int>(hw_threads) : 1;
#endif
}

SpectrumBaseMonitor::FftBackendPolicy parse_fft_backend_policy(
    const libconfig::Setting& settings,
    const char* key,
    const SpectrumBaseMonitor::FftBackendPolicy default_policy)
{
  std::string value;
  switch (default_policy)
  {
    case SpectrumBaseMonitor::FftBackendPolicy::Auto:
      value = "auto";
      break;
    case SpectrumBaseMonitor::FftBackendPolicy::Cpu:
      value = "cpu";
      break;
    case SpectrumBaseMonitor::FftBackendPolicy::Cuda:
      value = "cuda";
      break;
  }

  value = lowercase(jams::config_optional<std::string>(settings, key, value));
  if (value == "auto")
  {
    return SpectrumBaseMonitor::FftBackendPolicy::Auto;
  }
  if (value == "cpu")
  {
    return SpectrumBaseMonitor::FftBackendPolicy::Cpu;
  }
  if (value == "cuda" || value == "gpu")
  {
    return SpectrumBaseMonitor::FftBackendPolicy::Cuda;
  }
  throw std::runtime_error(std::string(key) + " must be one of: auto, cpu, cuda");
}

const char* backend_name(const SpectrumBaseMonitor::ActiveFftBackend backend)
{
  switch (backend)
  {
    case SpectrumBaseMonitor::ActiveFftBackend::Cpu:
      return "cpu";
    case SpectrumBaseMonitor::ActiveFftBackend::Cuda:
      return "cuda";
  }
  return "unknown";
}

std::size_t sk_time_series_required_bytes(
    const SpectrumBaseMonitor::CmplxStoredRingStorage::shape_type& shape)
{
  std::size_t elements = 1;
  for (const auto dim : shape)
  {
    if (dim != 0 && elements > std::numeric_limits<std::size_t>::max() / dim)
    {
      throw std::runtime_error("S(k,t) time-series size overflow");
    }
    elements *= dim;
  }
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(SpectrumBaseMonitor::CmplxStored))
  {
    throw std::runtime_error("S(k,t) time-series byte size overflow");
  }
  return elements * sizeof(SpectrumBaseMonitor::CmplxStored);
}

#if JAMS_HAS_FFTW_THREADS
std::mutex& fftw_threads_mutex()
{
  static std::mutex mutex;
  return mutex;
}

int& fftw_threads_refcount()
{
  static int refcount = 0;
  return refcount;
}

bool retain_fftw_thread_support()
{
  std::lock_guard<std::mutex> guard(fftw_threads_mutex());
  auto& refcount = fftw_threads_refcount();
  if (refcount == 0 && fftw_init_threads() == 0)
  {
    return false;
  }
  ++refcount;
  return true;
}

void release_fftw_thread_support()
{
  std::lock_guard<std::mutex> guard(fftw_threads_mutex());
  auto& refcount = fftw_threads_refcount();
  if (refcount <= 0)
  {
    return;
  }
  --refcount;
  if (refcount == 0)
  {
    fftw_cleanup_threads();
  }
}
#endif

}  // namespace

// ---------------------------------------------------------------------------
// Channel maps
// ---------------------------------------------------------------------------
SpectrumBaseMonitor::ChannelTransform SpectrumBaseMonitor::cartesian_channel_map()
{
  return ChannelTransform{};
}

SpectrumBaseMonitor::ChannelTransform SpectrumBaseMonitor::raise_lower_channel_map()
{
  ChannelTransform m;
  m.output_channels = 3;
  m.weights = jams::Mat<std::complex<double>, 3, 3>{
      kInvSqrtTwo, +kImagOne * kInvSqrtTwo, 0.0,
      kInvSqrtTwo, -kImagOne * kInvSqrtTwo, 0.0,
      0.0, 0.0, 1.0};
  m.use_local_frame = true;
  m.scale_to_physical_spin = true;
  return m;
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::configure_direct_sum_(
    const libconfig::Setting& settings,
    const KSamplingMode k_sampling_mode)
{
  if (!settings.exists("direct_sum"))
  {
    return;
  }

  const auto& direct_sum_settings = settings["direct_sum"];
  if (!direct_sum_settings.isGroup())
  {
    throw std::runtime_error("direct_sum must be a settings group");
  }
  if (!jams::config_optional<bool>(direct_sum_settings, "enabled", true))
  {
    return;
  }
  if (k_sampling_mode == KSamplingMode::FullGrid)
  {
    throw std::runtime_error("direct_sum is not compatible with full-grid k sampling");
  }
  if (!direct_sum_settings.exists("hkl_path"))
  {
    throw std::runtime_error("direct_sum.hkl_path is required when direct_sum.enabled is true");
  }

  spatial_transform_mode_ = SpatialTransformMode::DirectSum;
  direct_sum_backend_policy_ = parse_fft_backend_policy(
      direct_sum_settings, "backend", direct_sum_backend_policy_);

  if (direct_sum_settings.exists("window"))
  {
    const auto& window_settings = direct_sum_settings["window"];
    if (!window_settings.isGroup())
    {
      throw std::runtime_error("direct_sum.window must be a settings group");
    }
    const std::array<const char*, 3> axes {"x", "y", "z"};
    for (int axis = 0; axis < 3; ++axis)
    {
      if (!window_settings.exists(axes[axis]))
      {
        continue;
      }
      const auto& axis_settings = window_settings[axes[axis]];
      if (!axis_settings.isGroup())
      {
        throw std::runtime_error("direct_sum.window axis settings must be groups");
      }
      direct_sum_window_.enabled[axis] = true;
      if (axis_settings.exists("origin") && axis_settings.exists("center"))
      {
        throw std::runtime_error("direct_sum.window axis settings must specify at most one of origin and center");
      }
      if (axis_settings.exists("origin"))
      {
        direct_sum_window_.origin[axis] = jams::config_required<double>(axis_settings, "origin");
      }
      else if (axis_settings.exists("center"))
      {
        direct_sum_window_.origin[axis] = jams::config_required<double>(axis_settings, "center");
      }
      else
      {
        direct_sum_window_.default_origin[axis] = true;
      }
      direct_sum_window_.width[axis] = jams::config_required<double>(axis_settings, "width");
      if (direct_sum_window_.width[axis] <= 0.0)
      {
        throw std::runtime_error("direct_sum.window width must be greater than zero");
      }
    }
  }
}

void SpectrumBaseMonitor::configure_storage_backend_policy_(const libconfig::Setting& settings)
{
  std::string backend_setting = "auto";
  if (settings.exists("sk_time_series_backend"))
  {
    backend_setting = jams::config_required<std::string>(settings, "sk_time_series_backend");
  }
  else
  {
    // Backwards-compatible alias.
    backend_setting = jams::config_optional<std::string>(settings, "storage", backend_setting);
  }

  const auto backend_setting_lc = lowercase(backend_setting);
  if (backend_setting_lc == "auto")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::Auto;
    return;
  }
  if (backend_setting_lc == "memory" || backend_setting_lc == "in_memory")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::Memory;
    return;
  }
  if (backend_setting_lc == "file" || backend_setting_lc == "file_backed")
  {
    sk_time_series_backend_policy_ = SkTimeSeriesBackendPolicy::File;
    return;
  }

  throw std::runtime_error(
      "sk_time_series_backend must be one of: auto, memory, file");
}

void SpectrumBaseMonitor::configure_fft_backend_policy_(const libconfig::Setting& settings)
{
  if (use_direct_sum_())
  {
    spatial_fft_backend_policy_ = direct_sum_backend_policy_;
  }
  else
  {
    spatial_fft_backend_policy_ = parse_fft_backend_policy(
        settings, "spatial_fft_backend", spatial_fft_backend_policy_);
  }
  time_fft_backend_policy_ = parse_fft_backend_policy(
      settings, "time_fft_backend", time_fft_backend_policy_);
  cuda_time_fft_memory_limit_mib_ = jams::config_optional<int>(
      settings, "cuda_time_fft_memory_limit_mib", cuda_time_fft_memory_limit_mib_);
  if (cuda_time_fft_memory_limit_mib_ < 0)
  {
    throw std::runtime_error("cuda_time_fft_memory_limit_mib must be greater than or equal to zero");
  }

  const bool cuda_solver = globals::solver && globals::solver->is_cuda_solver();
  switch (spatial_fft_backend_policy_)
  {
    case FftBackendPolicy::Cpu:
      active_spatial_fft_backend_ = ActiveFftBackend::Cpu;
      break;
    case FftBackendPolicy::Auto:
#if HAS_CUDA
      active_spatial_fft_backend_ = cuda_solver ? ActiveFftBackend::Cuda : ActiveFftBackend::Cpu;
#else
      active_spatial_fft_backend_ = ActiveFftBackend::Cpu;
#endif
      break;
    case FftBackendPolicy::Cuda:
#if HAS_CUDA
      if (!cuda_solver)
      {
        throw std::runtime_error(
            use_direct_sum_()
                ? "direct_sum.backend = \"cuda\" requires a CUDA solver"
                : "spatial_fft_backend = \"cuda\" requires a CUDA solver");
      }
      active_spatial_fft_backend_ = ActiveFftBackend::Cuda;
#else
      throw std::runtime_error(
          use_direct_sum_()
              ? "direct_sum.backend = \"cuda\" requires a CUDA build"
              : "spatial_fft_backend = \"cuda\" requires a CUDA build");
#endif
      break;
  }

  if (time_fft_backend_policy_ == FftBackendPolicy::Cuda
      && active_spatial_fft_backend_ != ActiveFftBackend::Cuda)
  {
    throw std::runtime_error(
        use_direct_sum_()
            ? "time_fft_backend = \"cuda\" requires direct_sum.backend to select CUDA"
            : "time_fft_backend = \"cuda\" requires spatial_fft_backend to select CUDA");
  }
}

void SpectrumBaseMonitor::initialise_k_points_(
    const libconfig::Setting& settings,
    const KSamplingMode k_sampling_mode)
{
  KPointPathBuilder builder(*globals::lattice);
  const auto kspace_size = globals::lattice->kspace_size();

  std::cout << "  creating k-point list" << std::endl;
  if (k_sampling_mode == KSamplingMode::FullGrid)
  {
    builder.append_full_k_grid(k_points_, k_segment_offsets_, kspace_size);
    full_brillouin_zone_appended_ = true;
  }
  else if (use_direct_sum_())
  {
    auto& direct_sum_settings = const_cast<libconfig::Setting&>(settings["direct_sum"]);
    libconfig::Setting* points_per_segment = direct_sum_settings.exists("points_per_segment")
        ? &direct_sum_settings["points_per_segment"]
        : nullptr;
    builder.configure_exact_k_list(
        k_points_,
        k_segment_offsets_,
        direct_sum_settings["hkl_path"],
        points_per_segment);
    full_brillouin_zone_appended_ = false;
  }
  else
  {
    if (!settings.exists("hkl_path"))
    {
      throw std::runtime_error(
          "hkl_path is required unless direct_sum.enabled is true and direct_sum.hkl_path is set");
    }
    full_brillouin_zone_appended_ = builder.configure_k_list(
        k_points_, k_segment_offsets_, settings["hkl_path"], kspace_size);
  }
}

void SpectrumBaseMonitor::initialise_basis_phase_factors_()
{
  std::cout << "  generating basis phase factors" << std::endl;
  std::vector<jams::Vec<double, 3>> r_frac(num_basis_atoms());
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    r_frac[a] = globals::lattice->basis_site_atom(a).position_frac;
  }
  generate_phase_factors_(basis_phase_factors_, r_frac, k_points_);
}

double SpectrumBaseMonitor::direct_sum_window_weight_(
    const jams::Vec<double, 3>& position_cart) const
{
  double weight = 1.0;
  for (int axis = 0; axis < 3; ++axis)
  {
    if (!direct_sum_window_.enabled[axis])
    {
      continue;
    }
    const double width = direct_sum_window_.width[axis];
    const double start = direct_sum_window_.origin[axis] - 0.5 * width;
    const double x = (position_cart[axis] - start) / width;
    if (x < 0.0 || x > 1.0)
    {
      return 0.0;
    }
    weight *= fft_window_default_fraction(x);
  }
  return weight;
}

void SpectrumBaseMonitor::validate_direct_sum_window_extent_()
{
  bool has_window = false;
  for (const bool enabled : direct_sum_window_.enabled)
  {
    has_window = has_window || enabled;
  }
  if (!has_window)
  {
    return;
  }

  const auto& supercell = globals::lattice->get_supercell();
  const std::array<jams::Vec<double, 3>, 3> cell_edges {
      supercell.a1(),
      supercell.a2(),
      supercell.a3()};

  jams::Vec<double, 3> box_min {
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max()};
  jams::Vec<double, 3> box_max {
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest()};

  for (int corner_mask = 0; corner_mask < 8; ++corner_mask)
  {
    jams::Vec<double, 3> corner {0.0, 0.0, 0.0};
    for (int edge = 0; edge < 3; ++edge)
    {
      if ((corner_mask & (1 << edge)) != 0)
      {
        corner += cell_edges[edge];
      }
    }

    for (int axis = 0; axis < 3; ++axis)
    {
      box_min[axis] = std::min(box_min[axis], corner[axis]);
      box_max[axis] = std::max(box_max[axis], corner[axis]);
    }
  }

  const std::array<const char*, 3> axes {"x", "y", "z"};
  for (int axis = 0; axis < 3; ++axis)
  {
    if (!direct_sum_window_.enabled[axis])
    {
      continue;
    }
    if (direct_sum_window_.default_origin[axis])
    {
      direct_sum_window_.origin[axis] = 0.5 * (box_min[axis] + box_max[axis]);
      std::cout << "  direct_sum.window." << axes[axis]
                << " origin defaulted to simulation box center "
                << direct_sum_window_.origin[axis] << std::endl;
      direct_sum_window_.default_origin[axis] = false;
    }

    const double window_min = direct_sum_window_.origin[axis] - 0.5 * direct_sum_window_.width[axis];
    const double window_max = direct_sum_window_.origin[axis] + 0.5 * direct_sum_window_.width[axis];
    const double tolerance = 1.0e-10 * std::max({
        1.0,
        std::abs(window_min),
        std::abs(window_max),
        std::abs(box_min[axis]),
        std::abs(box_max[axis])});

    if (window_min < box_min[axis] - tolerance || window_max > box_max[axis] + tolerance)
    {
      std::ostringstream message;
      message << "direct_sum.window." << axes[axis]
              << " spans [" << window_min << ", " << window_max << "]"
              << " but the simulation box spans ["
              << box_min[axis] << ", " << box_max[axis] << "]"
              << " along Cartesian " << axes[axis]
              << "; the spatial window would be clipped by the simulation box";
      throw std::runtime_error(message.str());
    }
  }
}

void SpectrumBaseMonitor::initialise_direct_sum_sites_()
{
  if (!use_direct_sum_())
  {
    return;
  }

  direct_sum_sites_.clear();
  direct_sum_sites_.reserve(static_cast<std::size_t>(globals::num_spins));
  const auto kspace_size = globals::lattice->kspace_size();
  direct_sum_spatial_scale_ = 1.0 / std::sqrt(static_cast<double>(jams::product(kspace_size)));

  validate_direct_sum_window_extent_();

  for (int site = 0; site < globals::num_spins; ++site)
  {
    const double window_weight = direct_sum_window_weight_(
        globals::lattice->lattice_site_position_cart(site));
    if (window_weight == 0.0)
    {
      continue;
    }
    DirectSumSite direct_site;
    direct_site.site_index = site;
    direct_site.basis_index = static_cast<int>(globals::lattice->lattice_site_basis_index(site));
    direct_site.position_frac = globals::lattice->lattice_site_vector_frac(site);
    direct_site.window_weight = window_weight;
    direct_sum_sites_.push_back(direct_site);
  }

  std::cout << "  direct spatial summation sites "
            << direct_sum_sites_.size()
            << " / "
            << globals::num_spins
            << std::endl;
}

void SpectrumBaseMonitor::initialise_cuda_backend_()
{
#if HAS_CUDA
  if (active_spatial_fft_backend_ != ActiveFftBackend::Cuda || cuda_backend_)
  {
    return;
  }
  cuda_backend_ = make_cuda_backend_();
  std::cout << (use_direct_sum_() ? "  cuda direct spatial memory (MiB) "
                                  : "  cuda spatial FFT memory (MiB) ")
            << static_cast<double>(cuda_backend_->spatial_memory_bytes()) / (1024.0 * 1024.0)
            << std::endl;
#else
  if (active_spatial_fft_backend_ == ActiveFftBackend::Cuda)
  {
    throw std::runtime_error("CUDA spatial FFT backend selected in a non-CUDA build");
  }
#endif
}

SpectrumBaseMonitor::SpectrumBaseMonitor(
    const libconfig::Setting& settings,
    const KSamplingMode k_sampling_mode)
    : Monitor(settings),
      sk_time_series_(0, jams::instance().temp_directory_path())
{
  const auto kspace_size = globals::lattice->kspace_size();
  num_basis_atoms_ = globals::lattice->num_basis_sites();

  keep_negative_frequencies_ = jams::config_optional<bool>(
      settings,
      "keep_negative_frequencies",
      keep_negative_frequencies_);
  configure_direct_sum_(settings, k_sampling_mode);
  configure_storage_backend_policy_(settings);
  configure_fft_backend_policy_(settings);

  if (settings.exists("compute_periodogram"))
  {
    configure_periodogram(settings["compute_periodogram"]);
  }

  configure_fftw_threads_(settings);
  initialise_k_points_(settings, k_sampling_mode);
  initialise_basis_phase_factors_();
  initialise_direct_sum_sites_();
  initialise_cuda_backend_();
  log_fft_backend_info_();

  if (use_direct_sum_())
  {
    std::cout << "  direct spatial summation active; no sk_grid buffer needed" << std::endl;
  }
  else if (!use_cuda_spatial_fft_())
  {
    std::cout << "  allocating sk_grid buffer" << std::endl;
    zero(sk_grid_.resize(
        kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_basis_atoms_));
  }
  else
  {
    std::cout << "  deferring host sk_grid allocation because CUDA spatial FFT is active" << std::endl;
  }

  std::cout << "  deferring sk_time_series buffer allocation" << std::endl;
}

// ---------------------------------------------------------------------------
// Lifecycle and storage selection
// ---------------------------------------------------------------------------
SpectrumBaseMonitor::~SpectrumBaseMonitor()
{
  fftw_destroy_plan(sk_time_fft_plan_);
#if JAMS_HAS_FFTW_THREADS
  if (fftw_threads_enabled_)
  {
    release_fftw_thread_support();
  }
#endif
}

void SpectrumBaseMonitor::set_channel_map(const ChannelTransform& channel_map)
{
  if (channel_map.output_channels < 1 || channel_map.output_channels > 3)
  {
    throw std::runtime_error("SpectrumBaseMonitor::set_channel_map output_channels must be in [1,3]");
  }

  channel_transform_ = channel_map;

#if HAS_CUDA
  if (cuda_backend_)
  {
    cuda_backend_->reset_time_storage();
  }
#endif

  if (sk_time_series_storage_initialised_)
  {
    resize_channel_storage_();
  }
  periodogram_sample_index_ = 0;
  periodogram_window_count_ = 1;

  if (sk_time_fft_plan_)
  {
    fftw_destroy_plan(sk_time_fft_plan_);
    sk_time_fft_plan_ = nullptr;
  }
}

bool SpectrumBaseMonitor::needs_local_frame_mapping_() const
{
  return channel_transform_.use_local_frame;
}

bool SpectrumBaseMonitor::use_file_backed_sk_time_series_(const std::size_t required_bytes) const
{
  switch (sk_time_series_backend_policy_)
  {
    case SkTimeSeriesBackendPolicy::Auto:
      return required_bytes > kAutoFileBackedSkTimeSeriesThresholdBytes;
    case SkTimeSeriesBackendPolicy::Memory:
      return false;
    case SkTimeSeriesBackendPolicy::File:
      return true;
  }
  throw std::runtime_error("Invalid sk_time_series backend policy");
}

bool SpectrumBaseMonitor::use_cuda_spatial_fft_() const
{
  return active_spatial_fft_backend_ == ActiveFftBackend::Cuda;
}

bool SpectrumBaseMonitor::use_direct_sum_() const
{
  return spatial_transform_mode_ == SpatialTransformMode::DirectSum;
}

bool SpectrumBaseMonitor::use_cuda_time_fft_() const
{
  return active_time_fft_backend_ == ActiveFftBackend::Cuda;
}

bool SpectrumBaseMonitor::cuda_time_fft_requested_() const
{
  return time_fft_backend_policy_ == FftBackendPolicy::Cuda;
}

bool SpectrumBaseMonitor::cuda_time_fft_auto_() const
{
  return time_fft_backend_policy_ == FftBackendPolicy::Auto;
}

void SpectrumBaseMonitor::enable_cuda_time_fft_backend_()
{
  cuda_time_fft_supported_ = true;
  cuda_time_fft_needs_magnon_accumulation_ = true;
}

void SpectrumBaseMonitor::enable_cuda_frequency_slices_backend_()
{
  cuda_time_fft_supported_ = true;
  cuda_time_fft_needs_frequency_slices_ = true;
}

void SpectrumBaseMonitor::require_negative_frequencies_()
{
  if (sk_time_series_storage_initialised_)
  {
    throw std::runtime_error("negative frequency output must be required before spectrum storage is initialised");
  }
  keep_negative_frequencies_ = true;
}

void SpectrumBaseMonitor::validate_cuda_time_fft_backend_support_() const
{
  if (cuda_time_fft_requested_() && !cuda_time_fft_supported_)
  {
    throw std::runtime_error("time_fft_backend = \"cuda\" is not supported by this monitor");
  }
}

void SpectrumBaseMonitor::log_fft_backend_info_() const
{
  std::cout << "  spatial transform "
            << (use_direct_sum_() ? "direct-sum" : "fft-grid")
            << std::endl;
  std::cout << "  spatial backend " << backend_name(active_spatial_fft_backend_) << std::endl;
  std::cout << "  time FFT backend policy ";
  switch (time_fft_backend_policy_)
  {
    case FftBackendPolicy::Auto:
      std::cout << "auto";
      break;
    case FftBackendPolicy::Cpu:
      std::cout << "cpu";
      break;
    case FftBackendPolicy::Cuda:
      std::cout << "cuda";
      break;
  }
  std::cout << std::endl;
}

void SpectrumBaseMonitor::ensure_channel_storage_initialised_()
{
  if (!sk_time_series_storage_initialised_)
  {
    resize_channel_storage_();
  }
}

void SpectrumBaseMonitor::log_channel_storage_info_() const
{
  if (use_cuda_time_fft_())
  {
    std::cout << "    sk_time_series backend cuda device ring buffer" << std::endl;
    return;
  }

  const double sk_time_series_size_mib =
      static_cast<double>(sk_time_series_.required_bytes()) / (1024.0 * 1024.0);
  std::cout << "    sk_time_series size (MiB) " << sk_time_series_size_mib << std::endl;

  if (sk_time_series_.using_file_backed_ring_buffer())
  {
    std::cout << "    sk_time_series backend file-backed ring buffer "
              << sk_time_series_.file_path() << std::endl;
  }
  else
  {
    std::cout << "    sk_time_series backend in-memory" << std::endl;
  }
}

void SpectrumBaseMonitor::configure_cuda_time_fft_storage_()
{
  active_time_fft_backend_ = ActiveFftBackend::Cpu;

  if (!use_cuda_spatial_fft_() || time_fft_backend_policy_ == FftBackendPolicy::Cpu)
  {
    return;
  }

#if HAS_CUDA
  initialise_cuda_backend_();
  if (!cuda_time_fft_supported_)
  {
    if (cuda_time_fft_requested_())
    {
      throw std::runtime_error("time_fft_backend = \"cuda\" is not supported by this monitor");
    }
    if (cuda_time_fft_auto_())
    {
      std::cout << "  time FFT backend cpu" << std::endl;
      std::cout << "  CUDA time FFT is not supported by this monitor; using CPU time FFT" << std::endl;
    }
    return;
  }

  const int T = periodogram_props_.length;
  const int K = static_cast<int>(k_points_.size());
  const int A = num_basis_atoms_;
  const int C = num_channels();
  const int num_freq = num_frequencies();
  const bool use_multitaper = temporal_estimator_ == TemporalEstimator::Multitaper;
  const auto estimated_bytes = cuda_backend_->estimate_time_fft_memory_bytes(
      T,
      A,
      K,
      stored_channel_count_,
      C,
      num_freq,
      multitaper_count_,
      needs_local_frame_mapping_(),
      use_multitaper,
      cuda_time_fft_needs_magnon_accumulation_,
      cuda_time_fft_needs_frequency_slices_);

  bool use_cuda_time = false;
  if (cuda_time_fft_requested_())
  {
    use_cuda_time = true;
  }
  else if (cuda_time_fft_auto_())
  {
    std::size_t allowed_bytes = 0;
    if (cuda_time_fft_memory_limit_mib_ > 0)
    {
      allowed_bytes = static_cast<std::size_t>(cuda_time_fft_memory_limit_mib_) * 1024u * 1024u;
    }
    else
    {
      allowed_bytes = static_cast<std::size_t>(0.70 * static_cast<double>(cuda_backend_->free_device_memory_bytes()));
    }
    use_cuda_time = estimated_bytes <= allowed_bytes;
    std::cout << "  cuda time FFT estimated memory (MiB) "
              << static_cast<double>(estimated_bytes) / (1024.0 * 1024.0)
              << " allowed (MiB) "
              << static_cast<double>(allowed_bytes) / (1024.0 * 1024.0)
              << std::endl;
  }

  if (!use_cuda_time)
  {
    std::cout << "  time FFT backend cpu" << std::endl;
    std::cout << "  CUDA spatial FFT will copy compact k-path samples to CPU only" << std::endl;
    return;
  }

  cuda_backend_->configure_time_storage(
      T,
      A,
      K,
      stored_channel_count_,
      C,
      num_freq,
      keep_negative_frequencies_,
      needs_local_frame_mapping_(),
      cuda_time_fft_needs_magnon_accumulation_,
      cuda_time_fft_needs_frequency_slices_,
      use_multitaper);
  active_time_fft_backend_ = ActiveFftBackend::Cuda;
  std::cout << "  time FFT backend cuda" << std::endl;
#else
  if (cuda_time_fft_requested_())
  {
    throw std::runtime_error("time_fft_backend = \"cuda\" requires a CUDA build");
  }
#endif
}

void SpectrumBaseMonitor::resize_channel_storage_()
{
  const int T = periodogram_props_.length;
  const int K = static_cast<int>(k_points_.size());
  const int A = num_basis_atoms_;
  const int C = num_channels();

  if (needs_local_frame_mapping_())
  {
    stored_channel_count_ = 3;
  }
  else
  {
    stored_channel_count_ = C;
  }
  configure_cuda_time_fft_storage_();

  if (!use_cuda_time_fft_())
  {
    const CmplxStoredRingStorage::shape_type sk_shape{
        static_cast<std::size_t>(T),
        static_cast<std::size_t>(A),
        static_cast<std::size_t>(K),
        static_cast<std::size_t>(stored_channel_count_)};
    const auto required_bytes = sk_time_series_required_bytes(sk_shape);
    sk_time_series_.resize(
        sk_shape,
        use_file_backed_sk_time_series_(required_bytes));
  }
  sk_time_series_storage_initialised_ = true;
  log_channel_storage_info_();

  if (needs_local_frame_mapping_() && !use_cuda_time_fft_())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
    basis_mag_time_series_.zero();
  }
  else
  {
    basis_mag_time_series_.clear();
  }

  if (periodogram_window_.size() != T)
  {
    generate_normalised_window_(periodogram_window_, T);
  }

  if (temporal_estimator_ == TemporalEstimator::Multitaper)
  {
    if (multitaper_windows_.extent(0) != multitaper_count_
        || multitaper_windows_.extent(1) != T
        || multitaper_weights_.size() != static_cast<std::size_t>(multitaper_count_))
    {
      generate_normalised_dpss_tapers_(
          multitaper_windows_,
          multitaper_weights_,
          multitaper_count_,
          T,
          multitaper_bandwidth_);
    }
  }
}

// ---------------------------------------------------------------------------
// k-path configuration
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::configure_temporal_estimator_(libconfig::Setting& settings)
{
  std::string estimator = jams::config_optional<std::string>(settings, "estimator", "multitaper");
  estimator = lowercase(estimator);

  if (estimator == "welch")
  {
    temporal_estimator_ = TemporalEstimator::Welch;
    return;
  }

  if (estimator == "multitaper")
  {
    temporal_estimator_ = TemporalEstimator::Multitaper;

    if (settings.exists("multitaper_bandwidth_thz"))
    {
      throw std::runtime_error(
          "multitaper_bandwidth_thz has been removed; use dimensionless multitaper_bandwidth");
    }

    jams::require_mutually_exclusive_settings(
        settings, {"multitaper_bandwidth", "multitaper_time_bandwidth"});
    const bool has_time_bandwidth = settings.exists("multitaper_time_bandwidth");

    if (has_time_bandwidth)
    {
      // Backwards-compatible alias for older input files.
      multitaper_bandwidth_ = jams::config_required<double>(settings, "multitaper_time_bandwidth");
    }
    else
    {
      multitaper_bandwidth_ = jams::config_optional<double>(settings, "multitaper_bandwidth", multitaper_bandwidth_);
    }

    if (multitaper_bandwidth_ <= 0.0)
    {
      throw std::runtime_error("multitaper_bandwidth must be greater than zero");
    }
    if (multitaper_bandwidth_ >= static_cast<double>(periodogram_props_.length) / 2.0)
    {
      throw std::runtime_error(
          "multitaper bandwidth is too large for the configured periodogram length");
    }

    const int max_reasonable_tapers = static_cast<int>(std::floor(2.0 * multitaper_bandwidth_ - 1.0));
    if (max_reasonable_tapers < 1)
    {
      throw std::runtime_error("multitaper_bandwidth is too small for multitaper_tapers");
    }

    if (settings.exists("multitaper_tapers"))
    {
      multitaper_count_ = jams::config_required<int>(settings, "multitaper_tapers");
    }
    else
    {
      multitaper_count_ = max_reasonable_tapers;
    }

    if (multitaper_count_ <= 0)
    {
      throw std::runtime_error("multitaper_tapers must be greater than zero");
    }
    if (multitaper_count_ > periodogram_props_.length)
    {
      throw std::runtime_error("multitaper_tapers must be less than or equal to periodogram length");
    }
    if (multitaper_count_ > max_reasonable_tapers)
    {
      throw std::runtime_error("multitaper_tapers must satisfy multitaper_tapers <= floor(2 * multitaper_bandwidth - 1)");
    }
    return;
  }

  throw std::runtime_error("compute_periodogram.estimator must be either 'welch' or 'multitaper'");
}

void SpectrumBaseMonitor::configure_fftw_threads_(const libconfig::Setting& settings)
{
  fftw_thread_count_ = jams::config_optional<int>(settings, "fftw_threads", default_fftw_thread_count());
  if (fftw_thread_count_ < 1)
  {
    throw std::runtime_error("fftw_threads must be greater than or equal to 1");
  }

#if JAMS_HAS_FFTW_THREADS
  fftw_threads_enabled_ = retain_fftw_thread_support();
  if (!fftw_threads_enabled_)
  {
    if (fftw_thread_count_ > 1)
    {
      std::cout << "  fftw thread initialisation failed, using single-threaded FFTW" << std::endl;
    }
    fftw_thread_count_ = 1;
  }
#else
  if (fftw_thread_count_ > 1)
  {
    std::cout << "  fftw thread support unavailable, using single-threaded FFTW" << std::endl;
  }
  fftw_thread_count_ = 1;
#endif

  std::cout << "  fftw FFT threads " << fftw_thread_count_ << std::endl;
}

void SpectrumBaseMonitor::configure_periodogram(libconfig::Setting &settings)
{
  periodogram_props_.length = jams::config_required<int>(settings, "length");
  periodogram_props_.overlap = jams::config_optional<int>(settings, "overlap", periodogram_props_.length / 2);

  if (periodogram_props_.length <= 0)
  {
    throw std::runtime_error("Periodogram length must be greater than zero");
  }

  if (periodogram_props_.overlap < 0)
  {
    throw std::runtime_error("Periodogram overlap must be greater than or equal to zero");
  }

  if (periodogram_props_.overlap >= periodogram_props_.length)
  {
    throw std::runtime_error("Periodogram overlap must be less than periodogram length");
  }

  configure_temporal_estimator_(settings);
}

// ---------------------------------------------------------------------------
// Frequency-space processing
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::prepare_frequency_windows_()
{
  if (periodogram_window_.size() != periodogram_length())
  {
    generate_normalised_window_(periodogram_window_, periodogram_length());
  }
  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (multitaper_windows_.extent(0) != multitaper_count_
          || multitaper_windows_.extent(1) != periodogram_length()
          || multitaper_weights_.size() != static_cast<std::size_t>(multitaper_count_)))
  {
    generate_normalised_dpss_tapers_(
        multitaper_windows_,
        multitaper_weights_,
        multitaper_count_,
        periodogram_length(),
        multitaper_bandwidth_);
  }
}

const SpectrumBaseMonitor::CmplxMappedSlice& SpectrumBaseMonitor::compute_frequency_spectrum_at_k(
  const int kpoint_index)
{
  const int num_sites = num_basis_atoms();
  const int num_time_samples = periodogram_length();
  const int channels = num_channels();
  const bool use_local_frame = needs_local_frame_mapping_();

  prepare_frequency_windows_();

  if (use_cuda_time_fft_())
  {
#if HAS_CUDA
    if (!cuda_time_fft_needs_frequency_slices_)
    {
      throw std::runtime_error("CUDA frequency-slice time FFT is not enabled for this monitor");
    }
    assert(cuda_backend_);
    if (frequency_scratch_.extent(0) != num_sites
        || frequency_scratch_.extent(1) != num_time_samples
        || frequency_scratch_.extent(2) != channels)
    {
      frequency_scratch_.resize(num_sites, num_time_samples, channels);
    }
    cuda_backend_->configure_frequency_inputs(
        periodogram_window_,
        multitaper_windows_,
        multitaper_weights_,
        channel_transform_);
    cuda_backend_->compute_frequency_spectrum_at_k(
        kpoint_index,
        temporal_estimator_ == TemporalEstimator::Multitaper,
        multitaper_count_);
    cuda_backend_->copy_frequency_spectrum_slice_to_host(frequency_scratch_);
    return frequency_scratch_;
#else
    throw std::runtime_error("CUDA time FFT backend selected in a non-CUDA build");
#endif
  }

  if (!sk_time_fft_plan_
      || frequency_scratch_.extent(0) != num_sites
      || frequency_scratch_.extent(1) != num_time_samples
      || frequency_scratch_.extent(2) != channels)
  {
    if (sk_time_fft_plan_)
    {
      fftw_destroy_plan(sk_time_fft_plan_);
      sk_time_fft_plan_ = nullptr;
    }

    frequency_scratch_.resize(num_sites, num_time_samples, channels);

    const int n[1] = {num_time_samples};
    const int howmany = channels;
    const int istride = channels;
    const int ostride = channels;
    const int idist = 1;
    const int odist = 1;

    auto* dummy = FFTW_COMPLEX_CAST(&frequency_scratch_(0, 0, 0));

#if JAMS_HAS_FFTW_THREADS
    if (fftw_threads_enabled_)
    {
      std::lock_guard<std::mutex> guard(fftw_threads_mutex());
      fftw_plan_with_nthreads(fftw_thread_count_);
    }
#endif
    sk_time_fft_plan_ = fftw_plan_many_dft(
        1,
        n,
        howmany,
        dummy,
        nullptr,
        istride,
        idist,
        dummy,
        nullptr,
        ostride,
        odist,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    assert(sk_time_fft_plan_);
  }

  if (frequency_accum_.extent(0) != num_sites
      || frequency_accum_.extent(1) != num_time_samples
      || frequency_accum_.extent(2) != channels)
  {
    frequency_accum_.resize(num_sites, num_time_samples, channels);
  }

  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (frequency_taper_sum_.extent(0) != num_sites
          || frequency_taper_sum_.extent(1) != num_time_samples
          || frequency_taper_sum_.extent(2) != channels))
  {
    frequency_taper_sum_.resize(num_sites, num_time_samples, channels);
  }
  if (temporal_estimator_ == TemporalEstimator::Multitaper
      && (frequency_taper_power_sum_.extent(0) != num_sites
          || frequency_taper_power_sum_.extent(1) != num_time_samples
          || frequency_taper_power_sum_.extent(2) != channels))
  {
    frequency_taper_power_sum_.resize(num_sites, num_time_samples, channels);
  }
  const auto rotations = use_local_frame
      ? generate_sublattice_rotations_()
      : jams::MultiArray<jams::Mat<double, 3, 3>, 1>{};
  const auto* rotations_ptr = use_local_frame ? &rotations : nullptr;

  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      if (use_local_frame)
      {
        const jams::Vec<std::complex<double>, 3> spin_xyz = read_cartesian_spin_(a, t, kpoint_index);
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = map_spin_component_(a, c, spin_xyz, rotations_ptr);
        }
      }
      else
      {
        for (auto c = 0; c < channels; ++c)
        {
          const auto s = sk_time_series_(t, a, kpoint_index, c);
          frequency_scratch_(a, t, c) = jams::ComplexHi{s.real(), s.imag()};
        }
      }
    }
  }

  const double time_norm = 1.0 / static_cast<double>(num_time_samples);

  for (auto a = 0; a < num_sites; ++a)
  {
    std::vector<jams::ComplexHi> sk0(channels, jams::ComplexHi{0.0, 0.0});

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        sk0[c] += time_norm * frequency_scratch_(a, t, c);
      }
    }

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        frequency_accum_(a, t, c) = time_norm * (frequency_scratch_(a, t, c) - sk0[c]);
      }
    }
  }

  if (temporal_estimator_ == TemporalEstimator::Welch)
  {
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = periodogram_window_(t) * frequency_accum_(a, t, c);
        }
      }
      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);
    }
    return frequency_scratch_;
  }

  // Multitaper: average tapered complex spectra and preserve mean power.
  zero(frequency_taper_sum_);
  zero(frequency_taper_power_sum_);

  for (auto taper = 0; taper < multitaper_count_; ++taper)
  {
    const double taper_weight = multitaper_weights_(taper);
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = multitaper_windows_(taper, t) * frequency_accum_(a, t, c);
        }
      }

      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);

      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_taper_sum_(a, t, c) += taper_weight * frequency_scratch_(a, t, c);
          frequency_taper_power_sum_(a, t, c) += taper_weight * std::norm(frequency_scratch_(a, t, c));
        }
      }
    }
  }

  constexpr double kPhaseEpsilon = 1e-30;
  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        const auto mean_complex = frequency_taper_sum_(a, t, c);
        const double mean_power = std::max(0.0, frequency_taper_power_sum_(a, t, c));
        const double mean_abs = std::abs(mean_complex);
        if (mean_abs > kPhaseEpsilon)
        {
          frequency_scratch_(a, t, c) = (std::sqrt(mean_power) / mean_abs) * mean_complex;
        }
        else
        {
          frequency_scratch_(a, t, c) = jams::ComplexHi{0.0, 0.0};
        }
      }
    }
  }
  return frequency_scratch_;
}

void SpectrumBaseMonitor::for_each_frequency_spectrum_at_k(
    const int kpoint_index,
    const FrequencySpectrumCallback& callback)
{
  if (!callback)
  {
    return;
  }

  const int num_sites = num_basis_atoms();
  const int num_time_samples = periodogram_length();
  const int channels = num_channels();
  const bool use_local_frame = needs_local_frame_mapping_();

  prepare_frequency_windows_();

  if (use_cuda_time_fft_())
  {
#if HAS_CUDA
    if (!cuda_time_fft_needs_frequency_slices_)
    {
      throw std::runtime_error("CUDA frequency-slice time FFT is not enabled for this monitor");
    }
    assert(cuda_backend_);
    if (frequency_scratch_.extent(0) != num_sites
        || frequency_scratch_.extent(1) != num_time_samples
        || frequency_scratch_.extent(2) != channels)
    {
      frequency_scratch_.resize(num_sites, num_time_samples, channels);
    }
    cuda_backend_->configure_frequency_inputs(
        periodogram_window_,
        multitaper_windows_,
        multitaper_weights_,
        channel_transform_);

    if (temporal_estimator_ == TemporalEstimator::Welch)
    {
      cuda_backend_->compute_frequency_spectrum_at_k_window(kpoint_index, false, 0);
      cuda_backend_->copy_frequency_spectrum_slice_to_host(frequency_scratch_);
      callback(frequency_scratch_, 1.0);
      return;
    }

    for (auto taper = 0; taper < multitaper_count_; ++taper)
    {
      cuda_backend_->compute_frequency_spectrum_at_k_window(kpoint_index, true, taper);
      cuda_backend_->copy_frequency_spectrum_slice_to_host(frequency_scratch_);
      callback(frequency_scratch_, multitaper_weights_(taper));
    }
    return;
#else
    throw std::runtime_error("CUDA time FFT backend selected in a non-CUDA build");
#endif
  }

  if (!sk_time_fft_plan_
      || frequency_scratch_.extent(0) != num_sites
      || frequency_scratch_.extent(1) != num_time_samples
      || frequency_scratch_.extent(2) != channels)
  {
    if (sk_time_fft_plan_)
    {
      fftw_destroy_plan(sk_time_fft_plan_);
      sk_time_fft_plan_ = nullptr;
    }

    frequency_scratch_.resize(num_sites, num_time_samples, channels);

    const int n[1] = {num_time_samples};
    const int howmany = channels;
    const int istride = channels;
    const int ostride = channels;
    const int idist = 1;
    const int odist = 1;

    auto* dummy = FFTW_COMPLEX_CAST(&frequency_scratch_(0, 0, 0));

#if JAMS_HAS_FFTW_THREADS
    if (fftw_threads_enabled_)
    {
      std::lock_guard<std::mutex> guard(fftw_threads_mutex());
      fftw_plan_with_nthreads(fftw_thread_count_);
    }
#endif
    sk_time_fft_plan_ = fftw_plan_many_dft(
        1,
        n,
        howmany,
        dummy,
        nullptr,
        istride,
        idist,
        dummy,
        nullptr,
        ostride,
        odist,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    assert(sk_time_fft_plan_);
  }

  if (frequency_accum_.extent(0) != num_sites
      || frequency_accum_.extent(1) != num_time_samples
      || frequency_accum_.extent(2) != channels)
  {
    frequency_accum_.resize(num_sites, num_time_samples, channels);
  }

  const auto rotations = use_local_frame
      ? generate_sublattice_rotations_()
      : jams::MultiArray<jams::Mat<double, 3, 3>, 1>{};
  const auto* rotations_ptr = use_local_frame ? &rotations : nullptr;

  for (auto a = 0; a < num_sites; ++a)
  {
    for (auto t = 0; t < num_time_samples; ++t)
    {
      if (use_local_frame)
      {
        const jams::Vec<std::complex<double>, 3> spin_xyz = read_cartesian_spin_(a, t, kpoint_index);
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = map_spin_component_(a, c, spin_xyz, rotations_ptr);
        }
      }
      else
      {
        for (auto c = 0; c < channels; ++c)
        {
          const auto s = sk_time_series_(t, a, kpoint_index, c);
          frequency_scratch_(a, t, c) = jams::ComplexHi{s.real(), s.imag()};
        }
      }
    }
  }

  const double time_norm = 1.0 / static_cast<double>(num_time_samples);

  for (auto a = 0; a < num_sites; ++a)
  {
    std::vector<jams::ComplexHi> sk0(channels, jams::ComplexHi{0.0, 0.0});

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        sk0[c] += time_norm * frequency_scratch_(a, t, c);
      }
    }

    for (auto t = 0; t < num_time_samples; ++t)
    {
      for (auto c = 0; c < channels; ++c)
      {
        frequency_accum_(a, t, c) = time_norm * (frequency_scratch_(a, t, c) - sk0[c]);
      }
    }
  }

  if (temporal_estimator_ == TemporalEstimator::Welch)
  {
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = periodogram_window_(t) * frequency_accum_(a, t, c);
        }
      }
      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);
    }
    callback(frequency_scratch_, 1.0);
    return;
  }

  for (auto taper = 0; taper < multitaper_count_; ++taper)
  {
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto t = 0; t < num_time_samples; ++t)
      {
        for (auto c = 0; c < channels; ++c)
        {
          frequency_scratch_(a, t, c) = multitaper_windows_(taper, t) * frequency_accum_(a, t, c);
        }
      }

      auto* ptr = FFTW_COMPLEX_CAST(&frequency_scratch_(a, 0, 0));
      fftw_execute_dft(sk_time_fft_plan_, ptr, ptr);
    }
    callback(frequency_scratch_, multitaper_weights_(taper));
  }
}

bool SpectrumBaseMonitor::periodogram_window_complete() const
{
  return periodogram_sample_index_ >= periodogram_props_.length && periodogram_props_.length > 0;
}

bool SpectrumBaseMonitor::accumulate_magnon_spectrum_cuda(
    jams::MultiArray<jams::Vec<double, 3>, 2>& cumulative)
{
  if (!use_cuda_time_fft_())
  {
    return false;
  }

#if HAS_CUDA
  assert(cuda_backend_);
  prepare_frequency_windows_();

  cuda_backend_->configure_frequency_inputs(
      periodogram_window_,
      multitaper_windows_,
      multitaper_weights_,
      channel_transform_);
  cuda_backend_->accumulate_magnon_spectrum(
      periodogram_length(),
      num_basis_atoms(),
      num_k_points(),
      num_channels(),
      keep_negative_frequencies_,
      needs_local_frame_mapping_(),
      temporal_estimator_ == TemporalEstimator::Multitaper,
      multitaper_count_);
  cuda_backend_->copy_magnon_spectrum_to_host(cumulative);
  return true;
#else
  return false;
#endif
}

bool SpectrumBaseMonitor::accumulate_magnon_density_cuda(
    jams::MultiArray<double, 1>& cumulative)
{
  if (!use_cuda_time_fft_())
  {
    return false;
  }

#if HAS_CUDA
  assert(cuda_backend_);
  prepare_frequency_windows_();

  cuda_backend_->configure_frequency_inputs(
      periodogram_window_,
      multitaper_windows_,
      multitaper_weights_,
      channel_transform_);
  cuda_backend_->accumulate_magnon_density(
      periodogram_length(),
      num_basis_atoms(),
      num_k_points(),
      num_channels(),
      keep_negative_frequencies_,
      needs_local_frame_mapping_(),
      temporal_estimator_ == TemporalEstimator::Multitaper,
      multitaper_count_);
  cuda_backend_->copy_magnon_density_to_host(cumulative);
  return true;
#else
  return false;
#endif
}

void SpectrumBaseMonitor::advance_periodogram_window()
{
  const std::size_t overlap = static_cast<std::size_t>(periodogram_overlap());

  if (use_cuda_time_fft_())
  {
#if HAS_CUDA
    assert(cuda_backend_);
    cuda_backend_->advance_ring_window(periodogram_overlap());
#endif
    periodogram_sample_index_ = periodogram_props_.overlap;
    periodogram_window_count_++;
    return;
  }

  // Keep only the overlap tail of S(k,t) as the head of the next window.
  const std::size_t num_time = sk_time_series_.size(0);

  assert(overlap < num_time);
  sk_time_series_.advance_ring_window(overlap);

  if (needs_local_frame_mapping_())
  {
    // Keep overlap for per-sublattice magnetisation and clear the non-overlap region.
    const std::size_t num_sublattices = globals::lattice->num_basis_sites();
    const std::size_t num_period_samples = static_cast<std::size_t>(periodogram_length());
    if (num_period_samples > 0)
    {
      assert(overlap < num_period_samples);

      const std::size_t mag_source0 = num_period_samples - overlap;

      for (std::size_t sublattice = 0; sublattice < num_sublattices; ++sublattice)
      {
        if (overlap > 0)
        {
          auto* dst = &basis_mag_time_series_(sublattice, 0);
          const auto* src = &basis_mag_time_series_(sublattice, mag_source0);
          std::copy_n(src, overlap, dst);
        }
        std::fill_n(&basis_mag_time_series_(sublattice, overlap),
                    num_period_samples - overlap,
                    jams::Vec<double, 3>{0, 0, 0});
      }
    }
  }

  // Reset write index to the overlap boundary for the next incoming sample.
  periodogram_sample_index_ = periodogram_props_.overlap;
  periodogram_window_count_++;
}


// TODO: Remove this an refactor NeutronSpectrum to stream over k
const SpectrumBaseMonitor::CmplxMappedSpectrum& SpectrumBaseMonitor::finalise_periodogram_spectrum()
{
  const int num_sites = num_basis_atoms();
  const int num_k = num_k_points();
  const int channels = num_channels();
  const int num_freq_out = keep_negative_frequencies_ ? periodogram_length() : (periodogram_length() / 2 + 1);

  if (skw_buffer_.extent(0) != num_sites
      || skw_buffer_.extent(1) != num_freq_out
      || skw_buffer_.extent(2) != num_k
      || skw_buffer_.extent(3) != channels)
  {
    skw_buffer_.resize(num_sites, num_freq_out, num_k, channels);
  }

  for (auto k = 0; k < num_k; ++k)
  {
    auto& sw_spectrum = compute_frequency_spectrum_at_k(k);
    for (auto a = 0; a < num_sites; ++a)
    {
      for (auto f = 0; f < num_freq_out; ++f)
      {
        for (auto c = 0; c < channels; ++c)
        {
          skw_buffer_(a, f, k, c) = sw_spectrum(a, f, c);
        }
      }
    }
  }

  advance_periodogram_window();
  return skw_buffer_;
}

void SpectrumBaseMonitor::append_sk_sample_for_k_list(const jams::MultiArray<jams::Vec<std::complex<double>, 3>,4> &sk_sample,
                                                    const std::vector<jams::HKLIndex> &k_list)
{
  const auto time_index = static_cast<std::size_t>(periodogram_sample_index_);
  const auto num_basis = static_cast<std::size_t>(sk_sample.extent(3));
  const auto num_k = k_list.size();
  const auto stored_channels = static_cast<std::size_t>(stored_channel_count_);
  const bool use_local_frame = needs_local_frame_mapping_();
  std::vector<CmplxStored> tail_buffer(num_basis * num_k * stored_channels);

  for (auto a = 0; a < sk_sample.extent(3); ++a)
  {
    for (auto k = 0; k < k_list.size(); ++k)
    {
      if (!k_list[k].has_fft_index)
      {
        throw std::runtime_error("FFT spatial sampling received a k-point without an FFT grid index");
      }
      const auto [offset, is_conjugate] = k_list[k].index;
      const auto idx = offset;
      const auto base =
          (static_cast<std::size_t>(a) * num_k + static_cast<std::size_t>(k)) * stored_channels;

      jams::Vec<std::complex<double>, 3> spin_xyz;
      if (is_conjugate)
      {
        spin_xyz = basis_phase_factors_(a, k) * jams::conj(sk_sample(idx[0], idx[1], idx[2], a));
      }
      else
      {
        spin_xyz = basis_phase_factors_(a, k) * sk_sample(idx[0], idx[1], idx[2], a);
      }

      if (use_local_frame)
      {
        tail_buffer[base + 0] = CmplxStored{static_cast<float>(spin_xyz[0].real()), static_cast<float>(spin_xyz[0].imag())};
        tail_buffer[base + 1] = CmplxStored{static_cast<float>(spin_xyz[1].real()), static_cast<float>(spin_xyz[1].imag())};
        tail_buffer[base + 2] = CmplxStored{static_cast<float>(spin_xyz[2].real()), static_cast<float>(spin_xyz[2].imag())};
      }
      else
      {
        for (std::size_t c = 0; c < stored_channels; ++c)
        {
          const auto value = map_spin_component_(a, static_cast<int>(c), spin_xyz, nullptr);
          tail_buffer[base + c] = CmplxStored{static_cast<float>(value.real()), static_cast<float>(value.imag())};
        }
      }
    }
  }

  const std::array<std::size_t, 1> prefix{time_index};
  sk_time_series_.for_each_tail_block<3>(
      prefix,
      [&](CmplxStored* destination, const std::size_t logical_offset, const std::size_t count)
      {
        std::copy_n(tail_buffer.data() + logical_offset, count, destination);
      });
}

void SpectrumBaseMonitor::append_compact_sk_sample_(const std::vector<CmplxStored>& sample)
{
  const auto time_index = static_cast<std::size_t>(periodogram_sample_index_);
  const auto expected = static_cast<std::size_t>(num_basis_atoms_)
      * static_cast<std::size_t>(k_points_.size())
      * static_cast<std::size_t>(stored_channel_count_);
  if (sample.size() != expected)
  {
    throw std::runtime_error("CUDA compact S(k) sample has unexpected size");
  }

  const std::array<std::size_t, 1> prefix{time_index};
  sk_time_series_.for_each_tail_block<3>(
      prefix,
      [&](CmplxStored* destination, const std::size_t logical_offset, const std::size_t count)
      {
        std::copy_n(sample.data() + logical_offset, count, destination);
      });
}

void SpectrumBaseMonitor::store_direct_sum_snapshot_(
    const jams::MultiArray<double, 2>& spin_state)
{
  const auto num_basis = static_cast<std::size_t>(num_basis_atoms());
  const auto num_k = k_points_.size();
  const auto stored_channels = static_cast<std::size_t>(stored_channel_count_);
  const bool use_local_frame = needs_local_frame_mapping_();

  std::vector<jams::Vec<std::complex<double>, 3>> sk_sum(num_basis * num_k);
  for (auto& value : sk_sum)
  {
    value = {0.0, 0.0, 0.0};
  }

  for (const auto& site : direct_sum_sites_)
  {
    const jams::Vec<double, 3> spin = {
        spin_state(site.site_index, 0),
        spin_state(site.site_index, 1),
        spin_state(site.site_index, 2)};
    const double site_scale = direct_sum_spatial_scale_ * site.window_weight;

    for (std::size_t k = 0; k < num_k; ++k)
    {
      const auto phase = std::exp(-kImagTwoPi * jams::dot(k_points_[k].hkl, site.position_frac));
      auto& out = sk_sum[static_cast<std::size_t>(site.basis_index) * num_k + k];
      for (int c = 0; c < 3; ++c)
      {
        out[c] += (site_scale * spin[c]) * phase;
      }
    }
  }

  std::vector<CmplxStored> sample(num_basis * num_k * stored_channels);
  for (std::size_t a = 0; a < num_basis; ++a)
  {
    for (std::size_t k = 0; k < num_k; ++k)
    {
      const auto& spin_xyz = sk_sum[a * num_k + k];
      const auto base = (a * num_k + k) * stored_channels;
      if (use_local_frame)
      {
        sample[base + 0] = CmplxStored{static_cast<float>(spin_xyz[0].real()), static_cast<float>(spin_xyz[0].imag())};
        sample[base + 1] = CmplxStored{static_cast<float>(spin_xyz[1].real()), static_cast<float>(spin_xyz[1].imag())};
        sample[base + 2] = CmplxStored{static_cast<float>(spin_xyz[2].real()), static_cast<float>(spin_xyz[2].imag())};
      }
      else
      {
        for (std::size_t c = 0; c < stored_channels; ++c)
        {
          const auto mapped = map_spin_component_(
              static_cast<int>(a),
              static_cast<int>(c),
              spin_xyz,
              nullptr);
          sample[base + c] = CmplxStored{static_cast<float>(mapped.real()), static_cast<float>(mapped.imag())};
        }
      }
    }
  }

  append_compact_sk_sample_(sample);
}

// ---------------------------------------------------------------------------
// Local-frame mapping and accumulation
// ---------------------------------------------------------------------------
void SpectrumBaseMonitor::store_sublattice_magnetisation_(const jams::MultiArray<double, 2>& spin_state)
{
  if (basis_mag_time_series_.empty())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
    basis_mag_time_series_.zero();
  }
  const auto p = periodogram_sample_index();
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    jams::Vec<double, 3> spin = {spin_state(i, 0), spin_state(i, 1), spin_state(i, 2)};
    const auto m = globals::lattice->lattice_site_basis_index(i);
    basis_mag_time_series_(m, p) += spin;
  }
}

void SpectrumBaseMonitor::append_sublattice_magnetisation_sample_(
    const std::vector<jams::Vec<double, 3>>& basis_magnetisation)
{
  if (basis_magnetisation.size() != static_cast<std::size_t>(num_basis_atoms()))
  {
    throw std::runtime_error("CUDA sublattice magnetisation sample has unexpected size");
  }
  if (basis_mag_time_series_.empty())
  {
    basis_mag_time_series_.resize(num_basis_atoms(), periodogram_length());
    basis_mag_time_series_.zero();
  }
  const auto p = periodogram_sample_index();
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    basis_mag_time_series_(a, p) = basis_magnetisation[static_cast<std::size_t>(a)];
  }
}

jams::MultiArray<jams::Vec<double, 3>, 1> SpectrumBaseMonitor::compute_mean_basis_mag_directions_()
{
  if (periodogram_window_.empty())
  {
    generate_normalised_window_(periodogram_window_, periodogram_length());
  }

  jams::MultiArray<jams::Vec<double, 3>, 1> mean_directions(num_basis_atoms());
  zero(mean_directions);

  for (auto m = 0; m < num_basis_atoms(); ++m)
  {
    for (auto n = 0; n < periodogram_length(); ++n)
    {
      mean_directions(m) += periodogram_window_(n) * basis_mag_time_series_(m, n);
    }

    mean_directions(m) = jams::normalize(mean_directions(m));
  }

  return mean_directions;
}

jams::MultiArray<jams::Mat<double, 3, 3>, 1> SpectrumBaseMonitor::generate_sublattice_rotations_()
{
  jams::MultiArray<jams::Mat<double, 3, 3>, 1> rotations(num_basis_atoms());
  for (auto a = 0; a < num_basis_atoms(); ++a)
  {
    rotations(a) = kIdentityMat3;
  }

  if (!channel_transform_.use_local_frame)
  {
    return rotations;
  }

  const auto mean_directions = compute_mean_basis_mag_directions_();

  for (auto m = 0; m < mean_directions.size(); ++m)
  {
    jams::Vec<double, 3> n_hat = mean_directions(m);
    const double n_norm = jams::norm(n_hat);
    if (n_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    n_hat *= (1.0 / n_norm);

    const jams::Vec<double, 3> ex{1.0, 0.0, 0.0};
    const jams::Vec<double, 3> ey{0.0, 1.0, 0.0};
    const jams::Vec<double, 3> ez{0.0, 0.0, 1.0};

    const double ax = std::abs(jams::dot(ex, n_hat));
    const double ay = std::abs(jams::dot(ey, n_hat));
    const double az = std::abs(jams::dot(ez, n_hat));

    jams::Vec<double, 3> r = ex;
    double a_min = ax;
    if (ay < a_min)
    {
      r = ey;
      a_min = ay;
    }
    if (az < a_min)
    {
      r = ez;
      a_min = az;
    }

    jams::Vec<double, 3> e1 = r - jams::dot(r, n_hat) * n_hat;
    const double e1_norm = jams::norm(e1);
    if (e1_norm <= 0.0)
    {
      rotations(m) = kIdentityMat3;
      continue;
    }
    e1 *= (1.0 / e1_norm);

    jams::Vec<double, 3> e2 = jams::cross(n_hat, e1);

    jams::Mat<double, 3, 3> R = kIdentityMat3;
    R[0][0] = e1[0];    R[0][1] = e1[1];    R[0][2] = e1[2];
    R[1][0] = e2[0];    R[1][1] = e2[1];    R[1][2] = e2[2];
    R[2][0] = n_hat[0]; R[2][1] = n_hat[1]; R[2][2] = n_hat[2];

    rotations(m) = R;
  }

  return rotations;
}

jams::ComplexHi SpectrumBaseMonitor::map_spin_component_(
    const int basis_index,
    const int channel_index,
    const jams::Vec<std::complex<double>, 3>& spin_xyz,
    const jams::MultiArray<jams::Mat<double, 3, 3>, 1>* rotations) const
{
  jams::Vec<std::complex<double>, 3> s = spin_xyz;

  if (rotations)
  {
    s = (*rotations)(basis_index) * s;
  }

  if (channel_transform_.scale_to_physical_spin)
  {
    s *= basis_spin_length_(basis_index);
  }

  const auto& w = channel_transform_.weights[channel_index];
  return w[0] * s[0] + w[1] * s[1] + w[2] * s[2];
}

double SpectrumBaseMonitor::basis_spin_length_(const int basis_index) const
{
  if (basis_index < 0 || basis_index >= num_basis_atoms())
  {
    throw std::runtime_error("basis spin length requested for invalid basis index");
  }

  const auto material_index = globals::lattice->basis_site_atom(basis_index).material_index;
  const double moment = globals::lattice->material(material_index).moment;
  const double spin_length = moment / (kElectronGFactor * kBohrMagnetonIU);
  if (!std::isfinite(spin_length) || spin_length <= 0.0)
  {
    throw std::runtime_error("basis spin length must be finite and positive for spectrum monitors");
  }
  return spin_length;
}

jams::Vec<std::complex<double>, 3> SpectrumBaseMonitor::read_cartesian_spin_(const int basis_index,
                                                 const int time_index,
                                                 const int k_index) const
{
  assert(stored_channel_count_ >= 3);
  const auto sx = sk_time_series_(time_index, basis_index, k_index, 0);
  const auto sy = sk_time_series_(time_index, basis_index, k_index, 1);
  const auto sz = sk_time_series_(time_index, basis_index, k_index, 2);
  return jams::Vec<std::complex<double>, 3>{
      jams::ComplexHi{sx.real(), sx.imag()},
      jams::ComplexHi{sy.real(), sy.imag()},
      jams::ComplexHi{sz.real(), sz.imag()}
  };
}

void SpectrumBaseMonitor::generate_normalised_window_(jams::MultiArray<double, 1>& window, int num_time_samples)
{
  if (window.size() != num_time_samples)
  {
    window.resize(num_time_samples);
  }

  double w2sum = 0.0;
  for (int i = 0; i < num_time_samples; ++i)
  {
    const double w = fft_window_default(i, num_time_samples);
    window(i) = w;
    w2sum += w * w;
  }

  const double inv_rms =
      (w2sum > 0.0) ? 1.0 / std::sqrt(w2sum / static_cast<double>(num_time_samples)) : 1.0;

  for (int i = 0; i < num_time_samples; ++i)
  {
    window(i) *= inv_rms;
  }
}

void SpectrumBaseMonitor::generate_normalised_dpss_tapers_(
    jams::MultiArray<double, 2>& tapers,
    jams::MultiArray<double, 1>& taper_weights,
    const int num_tapers,
    const int num_time_samples,
    const double time_bandwidth)
{
  if (num_tapers <= 0)
  {
    throw std::runtime_error("num_tapers must be greater than zero");
  }
  if (num_time_samples <= 0)
  {
    throw std::runtime_error("num_time_samples must be greater than zero");
  }
  if (num_tapers > num_time_samples)
  {
    throw std::runtime_error("num_tapers must be less than or equal to num_time_samples");
  }
  if (time_bandwidth <= 0.0)
  {
    throw std::runtime_error("time_bandwidth must be greater than zero");
  }
  if (time_bandwidth >= static_cast<double>(num_time_samples) / 2.0)
  {
    throw std::runtime_error("time_bandwidth must be less than num_time_samples / 2");
  }

  if (tapers.extent(0) != num_tapers || tapers.extent(1) != num_time_samples)
  {
    tapers.resize(num_tapers, num_time_samples);
  }
  if (taper_weights.size() != static_cast<std::size_t>(num_tapers))
  {
    taper_weights.resize(num_tapers);
  }

  const int N = num_time_samples;
  const int K = num_tapers;
  const double W = time_bandwidth / static_cast<double>(N);
  const double cos_2piW = std::cos(kTwoPi * W);

  std::vector<double> diag(static_cast<std::size_t>(N), 0.0);
  std::vector<double> off(static_cast<std::size_t>(std::max(1, N - 1)), 0.0);
  for (int n = 0; n < N; ++n)
  {
    const double x = 0.5 * static_cast<double>(N - 1 - 2 * n);
    diag[n] = x * x * cos_2piW;
  }
  for (int n = 0; n < N - 1; ++n)
  {
    const auto m = static_cast<double>(n + 1);
    off[n] = 0.5 * m * static_cast<double>(N - (n + 1));
  }

  std::vector<double> eigenvectors;
  std::vector<double> eigenvalues;
  jams::solve_symmetric_tridiagonal_top_eigenvectors(diag, off, eigenvectors, eigenvalues, N, K);

  std::vector<double> ordered_eigenvalues(static_cast<std::size_t>(K), 0.0);
  double eigenvalue_sum = 0.0;
  for (int out_k = 0; out_k < K; ++out_k)
  {
    const int src_k = K - 1 - out_k;
    const double lambda = std::max(0.0, eigenvalues[static_cast<std::size_t>(src_k)]);
    ordered_eigenvalues[static_cast<std::size_t>(out_k)] = lambda;
    eigenvalue_sum += lambda;
  }
  if (eigenvalue_sum > 0.0)
  {
    const double inv_sum = 1.0 / eigenvalue_sum;
    for (int out_k = 0; out_k < K; ++out_k)
    {
      taper_weights(out_k) = ordered_eigenvalues[static_cast<std::size_t>(out_k)] * inv_sum;
    }
  }
  else
  {
    const double uniform_weight = 1.0 / static_cast<double>(K);
    for (int out_k = 0; out_k < K; ++out_k)
    {
      taper_weights(out_k) = uniform_weight;
    }
  }

  const double rms_scale = std::sqrt(static_cast<double>(N)); // unit RMS as used for Welch window
  for (int out_k = 0; out_k < K; ++out_k)
  {
    // Selected eigenpairs are returned in ascending order; pick descending.
    const int src_k = K - 1 - out_k;

    // Deterministic sign convention for reproducibility.
    double sign = 1.0;
    for (int n = 0; n < N; ++n)
    {
      const double v = eigenvectors[static_cast<std::size_t>(n) + static_cast<std::size_t>(src_k) * static_cast<std::size_t>(N)];
      if (std::abs(v) > 1e-14)
      {
        sign = (v >= 0.0) ? 1.0 : -1.0;
        break;
      }
    }

    for (int n = 0; n < N; ++n)
    {
      const double v = eigenvectors[static_cast<std::size_t>(n) + static_cast<std::size_t>(src_k) * static_cast<std::size_t>(N)];
      tapers(out_k, n) = sign * rms_scale * v;
    }
  }
}

void SpectrumBaseMonitor::generate_phase_factors_(
    jams::MultiArray<jams::ComplexHi, 2>& phase_factors,
    const std::vector<jams::Vec<double, 3>>& r_frac,
    const std::vector<jams::HKLIndex>& kpoints)
{
  if (phase_factors.extent(0) != r_frac.size() || phase_factors.extent(1) != kpoints.size())
  {
    phase_factors.resize(r_frac.size(), kpoints.size());
  }

  for (auto a = 0; a < r_frac.size(); ++a)
  {
    const auto& r = r_frac[a];
    for (auto k = 0; k < kpoints.size(); ++k)
    {
      const auto& q = kpoints[k].hkl;
      phase_factors(a, k) = exp(-kImagTwoPi * jams::dot(q, r));
    }
  }
}

void SpectrumBaseMonitor::store_sk_snapshot(const jams::MultiArray<double, 2> &data)
{
  ensure_channel_storage_initialised_();

  if (use_direct_sum_())
  {
    if (needs_local_frame_mapping_() && !use_cuda_spatial_fft_())
    {
      store_sublattice_magnetisation_(data);
    }

    if (use_cuda_spatial_fft_())
    {
#if HAS_CUDA
      initialise_cuda_backend_();
      std::vector<CmplxStored> compact_sample;
      std::vector<jams::Vec<double, 3>> basis_magnetisation;
      const bool copy_sample_to_host = !use_cuda_time_fft_();
      const bool copy_basis_magnetisation_to_host =
          copy_sample_to_host && needs_local_frame_mapping_();

      cuda_backend_->store_sample(
          data,
          periodogram_sample_index_,
          stored_channel_count_,
          needs_local_frame_mapping_(),
          channel_transform_,
          copy_sample_to_host,
          compact_sample,
          copy_basis_magnetisation_to_host,
          basis_magnetisation);

      if (copy_sample_to_host)
      {
        append_compact_sk_sample_(compact_sample);
        if (copy_basis_magnetisation_to_host)
        {
          append_sublattice_magnetisation_sample_(basis_magnetisation);
        }
      }
      periodogram_sample_index_++;
      return;
#else
      throw std::runtime_error("CUDA direct spatial backend selected in a non-CUDA build");
#endif
    }

    store_direct_sum_snapshot_(data);
    periodogram_sample_index_++;
    return;
  }

  if (use_cuda_spatial_fft_())
  {
#if HAS_CUDA
    initialise_cuda_backend_();
    std::vector<CmplxStored> compact_sample;
    std::vector<jams::Vec<double, 3>> basis_magnetisation;
    const bool copy_sample_to_host = !use_cuda_time_fft_();
    const bool copy_basis_magnetisation_to_host =
        copy_sample_to_host && needs_local_frame_mapping_();

    cuda_backend_->store_sample(
        data,
        periodogram_sample_index_,
        stored_channel_count_,
        needs_local_frame_mapping_(),
        channel_transform_,
        copy_sample_to_host,
        compact_sample,
        copy_basis_magnetisation_to_host,
        basis_magnetisation);

    if (copy_sample_to_host)
    {
      append_compact_sk_sample_(compact_sample);
      if (copy_basis_magnetisation_to_host)
      {
        append_sublattice_magnetisation_sample_(basis_magnetisation);
      }
    }
    periodogram_sample_index_++;
    return;
#else
    throw std::runtime_error("CUDA spatial FFT backend selected in a non-CUDA build");
#endif
  }

  fft_lattice_vector_field_to_kspace(
      data,
      sk_grid_,
      *globals::lattice,
      fftw_thread_count_);

  if (needs_local_frame_mapping_())
  {
    store_sublattice_magnetisation_(data);
  }
  append_sk_sample_for_k_list(sk_grid_, k_points_);
  periodogram_sample_index_++;
}

void SpectrumBaseMonitor::print_info() const
{
  std::cout << "\n";
  std::cout << "  number of samples " << periodogram_length() << "\n";
  std::cout << "  sampling time (ps) " << sample_time_interval() << "\n";
  std::cout << "  acquisition time (ps) " << sample_time_interval() * periodogram_length() << "\n";
  std::cout << "  frequency resolution (THz) " << frequency_resolution_thz() << "\n";
  std::cout << "  maximum frequency (THz) " << max_frequency_thz() << "\n";
  std::cout << "  channels " << num_channels() << "\n";
  if (temporal_estimator_ == TemporalEstimator::Welch)
  {
    std::cout << "  temporal estimator welch\n";
  }
  else
  {
    std::cout << "  temporal estimator multitaper\n";
    std::cout << "  multitaper tapers " << multitaper_count_ << "\n";
    std::cout << "  multitaper bandwidth (NW) " << multitaper_bandwidth_ << "\n";
  }
  std::cout << "\n";
}
