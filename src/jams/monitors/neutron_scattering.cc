// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/fft.h"
#include "neutron_scattering.h"

using namespace std;
using namespace jams;
using Complex = std::complex<double>;

class Solver;

/**
 * Maps an index for an FFTW ordered array into the correct location and conjugation
 *
 * FFTW real to complex (r2c) transforms only give N/2+1 outputs in the last dimensions
 * as all other values can be calculated by Hermitian symmetry. This function maps
 * the 3D array position of a set of general indicies (positive, negative, larger than
 * the fft_size) to the correct location and sets a bool as to whether the value must
 * be conjugated
 *
 * @tparam T
 * @param k general 3D indicies (positive, negative, larger than fft_size)
 * @param N logical dimensions of the fft
 * @return pair {T: remapped index, bool: do conjugate}
 */
HKLIndex NeutronScatteringMonitor::fftw_remap_index_real_to_complex(Vec<int,3> k, const Vec<int,3> &N) {
  Vec3 hkl = scale(k, 1.0/to_double(N));
  Vec3 xyz = lattice->get_unitcell().inverse_matrix() * hkl;

  auto index = (k % N + N) % N;

  if (index.back() < N.back()/2+1) {
    // return normal index
    return HKLIndex{hkl, xyz, index, false};
  } else {
    // return Hermitian conjugate
    index = (-k % N + N) % N;
    return HKLIndex{hkl, xyz, index, true};
  }
}

/**
 * Compute the alpha, beta components of the scattiner cross section.
 *
 * @param alpha cartesian component {x,y,z}
 * @param beta  cartesian component {x,y,z}
 * @param site_a unit cell site index
 * @param site_b unit cell site index
 * @param sqw_a spin data for site a in reciprocal space and frequency space
 * @param sqw_b spin data for site b in reciprocal space and frequency space
 * @param hkl_indicies list of reciprocal space hkl indicies
 * @return
 */
MultiArray<Complex, 2> NeutronScatteringMonitor::partial_cross_section(const int alpha, const int beta, const unsigned site_a, const unsigned site_b) {
  const auto num_freqencies = sqw_.size(0);
  const auto num_reciprocal_points = sqw_.size(1);
  const Vec3 r_frac = lattice->motif_atom(site_b).pos - lattice->motif_atom(site_a).pos;

  // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
  MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);

  for (auto k = 0; k < num_reciprocal_points; ++k) {
    const auto q_frac = path_[k].hkl;
    const auto Q = zero_safe_normalize(path_[k].xyz);
    const auto phase = exp(-kImagTwoPi * dot(q_frac, r_frac));

    for (auto f = 0; f < num_freqencies; ++f) {
      convolved(f, k) = (kronecker_delta(alpha, beta) - Q[alpha]*Q[beta])
                        * phase * conj(sqw_(f, k, site_a, alpha)) * sqw_(f, k, site_b, beta);
    }
  }

  return convolved;
}

/**
 * Generate a path between nodes in reciprocal space sampling the kspace discretely.
 *
 * @param hkl_nodes
 * @param reciprocal_space_size
 * @return
 */
vector<HKLIndex> NeutronScatteringMonitor::generate_hkl_reciprocal_space_path(const vector<Vec3> &hkl_nodes, const Vec3i &reciprocal_space_size) {
  vector<HKLIndex> hkl_path;
  for (auto n = 0; n < hkl_nodes.size()-1; ++n) {
    Vec3i origin = to_int(scale(hkl_nodes[n], reciprocal_space_size));
    Vec3i displacement = to_int(scale(hkl_nodes[n+1], reciprocal_space_size)) - origin;
    Vec3i delta = normalize_components(displacement);

    // use +1 to include the last point on the displacement
    const auto num_coordinates = abs_max(displacement) + 1;

    Vec3i coordinate = origin;
    for (auto i = 0; i < num_coordinates; ++i) {
      // map an arbitrary coordinate into the limited k indicies of the reduced brillouin zone
      hkl_path.push_back(
          fftw_remap_index_real_to_complex(coordinate, reciprocal_space_size));

      coordinate += delta;
    }
  }
  // remove duplicates in the path where start and end indicies are the same at nodes
  hkl_path.erase(std::unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());

  return hkl_path;
}

NeutronScatteringMonitor::NeutronScatteringMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  time_point_counter_ = 0;

  libconfig::Setting &solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_ * t_step;
  int num_samples = ceil(t_run / t_sample);
  double freq_max = 1.0 / (2.0 * t_sample);
  freq_delta_ = 1.0 / (num_samples * t_sample);

  auto kspace_size = lattice->kspace_size();
  auto num_sites = lattice->num_motif_atoms();

  cout << "\n";
  cout << "  number of samples " << num_samples << "\n";
  cout << "  sampling time (s) " << t_sample << "\n";
  cout << "  acquisition time (s) " << t_sample * num_samples << "\n";
  cout << "  frequency resolution (THz) " << freq_delta_ / kTHz << "\n";
  cout << "  maximum frequency (THz) " << freq_max / kTHz << "\n";
  cout << "\n";

  auto& cfg_nodes = settings["hkl_path"];

  vector<Vec3> hkl_path_nodes(cfg_nodes.getLength());
  for (auto i = 0; i < cfg_nodes.getLength(); ++i) {
    hkl_path_nodes[i] = Vec3{cfg_nodes[i][0], cfg_nodes[i][1], cfg_nodes[i][2]};
  }

  path_ = generate_hkl_reciprocal_space_path(hkl_path_nodes, kspace_size);

  sq_.resize(kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_sites, 3);
  sqw_.resize(num_samples, path_.size(), num_sites, 3);
  sqw_.zero();

  /**
   * @warning FFTW_PRESERVE_INPUT is not supported for r2c arrays
   * http://www.fftw.org/doc/One_002dDimensional-DFTs-of-Real-Data.html
   */
  {
    auto s_backup = globals::s;

    fft_plan_to_qspace_ =
        fft_plan_transform_to_reciprocal_space(globals::s.data(), sq_.data(), kspace_size, num_sites);

    globals::s = s_backup;
  }
}

void NeutronScatteringMonitor::update(Solver * solver) {
  using namespace globals;
  assert(fft_plan_to_qspace_ != nullptr);
  assert(sq_.elements() > 0);

  const auto kspace_size = lattice->kspace_size();
  const auto num_sites = ::lattice->num_motif_atoms();

  /**
   * @warning fftw_execute doesn't call s.data() so the CPU and GPU memory don't sync
   * we must force them to sync manually
   */
  force_multiarray_sync(globals::s);

  fftw_execute(fft_plan_to_qspace_);

  std::transform(sq_.begin(), sq_.end(), sq_.begin(),
      [kspace_size](const Complex &a) { return a / sqrt(product(kspace_size)); });


  // store data just on the path of interest
  // extra safety in case there is an extra one time point due to floating point maths
  if (time_point_counter_ < sqw_.size(0)) {
    for (auto k = 0; k < path_.size(); ++k) {
      auto idx = path_[k].index;
      for (auto site = 0; site < num_sites; ++site) {
        for (auto n = 0; n < 3; ++n) {
          if (path_[k].conjugate) {
            sqw_(time_point_counter_, k, site, n) = conj(sq_(idx[0], idx[1], idx[2], site, n));
          } else {
            sqw_(time_point_counter_, k, site, n) = sq_(idx[0], idx[1], idx[2], site, n);
          }
        }
      }
    }
  }

  time_point_counter_++;
}

fftw_plan NeutronScatteringMonitor::fft_plan_transform_to_reciprocal_space(double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & num_sites) {
  assert(rspace != nullptr);
  assert(kspace != nullptr);
  assert(sum(kspace_size) > 0);

  int rank              = 3;
  int transform_size[3] = {kspace_size[0], kspace_size[1], kspace_size[2]};
  int num_transforms    = 3 * num_sites;
  int *nembed           = nullptr;
  int stride            = 3 * num_sites;
  int dist              = 1;

  /**
   * @warning FFTW_PRESERVE_INPUT is not supported for r2c arrays
   * http://www.fftw.org/doc/One_002dDimensional-DFTs-of-Real-Data.html
   */
  return fftw_plan_many_dft_r2c(
      rank, transform_size, num_transforms,
      rspace, nembed, stride, dist,
      FFTWCAST(kspace), nembed, stride, dist,
      FFTW_MEASURE);
}

void NeutronScatteringMonitor::fft_to_frequency() {

  const int num_time_samples = sqw_.size(0);
  const int num_space_samples = sqw_.size(1);
  const int num_sites = sqw_.size(2);

  // window the data and normalize
  for (auto i = 0; i < num_time_samples; ++i) {
    for (auto j = 0; j < num_space_samples; ++j) {
      for (auto m = 0; m < num_sites; ++m) {
        for (auto n = 0; n < 3; ++n) {
          sqw_(i,j,m,n) *= fft_window_default(i, num_time_samples) / double(num_time_samples);
        }
      }
    }
  }

  int rank              = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms    = num_space_samples * num_sites * 3;
  int nembed[]          = {num_time_samples};
  int stride            = num_space_samples * num_sites * 3;
  int dist              = 1;

  fftw_plan fft_plan = fftw_plan_many_dft(
      rank,transform_size,num_transforms,
      FFTWCAST(sqw_.data()),nembed,stride,dist,
      FFTWCAST(sqw_.data()),nembed,stride,dist,
      FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

  fftw_execute(fft_plan);
  fftw_destroy_plan(fft_plan);
}

void NeutronScatteringMonitor::post_process() {
  const auto time_points = sqw_.size(0);
  const auto space_points = sqw_.size(1);

  jams::MultiArray<Complex,2> total_cross_section(time_points, space_points);
  total_cross_section.zero();

  fft_to_frequency();

  for (auto site_a = 0; site_a < ::lattice->num_motif_atoms(); ++site_a) {
    for (auto site_b = 0; site_b < ::lattice->num_motif_atoms(); ++site_b) {
      // loop xx, xy, ... yz zz
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          auto cross_section = partial_cross_section(i, j, site_a, site_b);
          std::transform(total_cross_section.begin(), total_cross_section.end(),
                         cross_section.begin(), total_cross_section.begin(), std::plus<Complex>());
        }
      }
    }
  }

  std::ofstream sqwfile(seedname + "_sqw.tsv");

  sqwfile << "index\t";
  sqwfile << "h\t";
  sqwfile << "k\t";
  sqwfile << "l\t";
  sqwfile << "qx\t";
  sqwfile << "qy\t";
  sqwfile << "qz\t";
  sqwfile << "freq_THz\t";
  sqwfile << "energy_meV\t";
  sqwfile << "sigma_re\t";
  sqwfile << "sigma_im\n";

  auto format_int = [](ostream& os) -> ostream& {
      return os << std::fixed; };

  auto format_fix = [](ostream& os) -> ostream& {
      return os << std::setprecision(8) << std::setw(12) << std::fixed; };

  auto format_sci = [](ostream& os) -> ostream& {
      return os << std::setprecision(8) << std::setw(12) << std::scientific; };

  for (auto i = 0; i < (time_points/2) + 1; ++i) {
    for (auto j = 0; j < path_.size(); ++j) {
      sqwfile << format_int << j << "\t";
      sqwfile << format_fix << path_[j].hkl << "\t";
      sqwfile << format_fix << path_[j].xyz << "\t";
      sqwfile << format_fix << (i*freq_delta_ / 1e12) << "\t"; // THz
      sqwfile << format_fix << (i*freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
      sqwfile << format_sci << total_cross_section(i,j).real() << "\t";
      sqwfile << format_sci << total_cross_section(i,j).imag() << "\n";
    }
    sqwfile << std::endl;
  }

  sqwfile.close();
}

NeutronScatteringMonitor::~NeutronScatteringMonitor() {
  if (fft_plan_to_qspace_) {
    fftw_destroy_plan(fft_plan_to_qspace_);
    fft_plan_to_qspace_ = nullptr;
  }
}
