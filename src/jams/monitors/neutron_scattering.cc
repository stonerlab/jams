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
#include "jams/interface/fft.h"
#include "neutron_scattering.h"

using namespace std;
using namespace jams;
using Complex = std::complex<double>;

class Solver;

struct FormFactorParams {
    double A, a, B, b, C, c, D; };

struct FormFactorG {
    double g0, g2, g4, g6; };

struct FormFactorJ {
    FormFactorParams j0, j2, j4, j6; };

namespace jams {
    template<>
    inline FormFactorParams config_required(const libconfig::Setting &s, const std::string &name) {
      return {double{s[name][0]}, double{s[name][1]}, double{s[name][2]}, double{s[name][3]},
              double{s[name][4]}, double{s[name][5]}, double{s[name][6]}};
    }

    template<>
    inline FormFactorG config_required(const libconfig::Setting &s, const std::string &name) {
      return {double{s[name][0]}, double{s[name][1]}, double{s[name][2]}, double{s[name][3]}};
    }
}

double form_factor_jl(const int& l, const double& s, const FormFactorParams& f) {
  if (f.A == 0.0 && f.B == 0.0 && f.C == 0.0 && f.D == 0.0) return 0.0;

  double p, s2 = s * s;
  (l == 0) ? p = 1.0 : p = s2;

  return f.A * p * exp(-f.a * s2) + f.B * p * exp(-f.b * s2) + f.C * p * exp(-f.c * s2) + f.D * p;
}

double form_factor_q(const Vec3& q, const FormFactorG& g, const FormFactorJ& j) {
  auto s = norm(q) / (4.0 * kPi * kMeterToAngstroms * lattice->parameter());

  return  0.5 * ( g.g0 * form_factor_jl(0, s, j.j0) + g.g2 * form_factor_jl(2, s, j.j2)
                + g.g4 * form_factor_jl(4, s, j.j4) + g.g6 * form_factor_jl(6, s, j.j6) );
}

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
HKLIndex NeutronScatteringMonitor::fftw_remap_index_real_to_complex(Vec3i k, const Vec3i &N) {
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
MultiArray<Complex, 2> NeutronScatteringMonitor::compute_unpolarized_cross_section(const jams::MultiArray<Vec<Complex,3>, 3>& spectrum) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = spectrum.size(1);
  const auto num_sites = ::lattice->num_motif_atoms();

  MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);
  convolved.zero();
  for (auto a = 0; a < num_sites; ++a) {
    for (auto b = 0; b < num_sites; ++b) {
      Vec3 r = lattice->motif_atom(b).fractional_pos - lattice->motif_atom(a).fractional_pos;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        auto q = paths_[k].hkl;
        auto Q = unit_vector(paths_[k].xyz);
        auto prefactor = exp(-kImagTwoPi * dot(q, r)) * form_factors_(k, a) * form_factors_(k, b);
        for (auto f = 0; f < num_freqencies; ++f) {
          for (auto i : {0,1,2}) {
            for (auto j : {0,1,2}) {
              convolved(f, k) += prefactor * (kronecker_delta(i, j) - Q[i] * Q[j]) * conj(spectrum(f, k, a)[i]) * spectrum(f, k, b)[j];
            }
          }
        }
      }
    }
  }
  return convolved;
}

jams::MultiArray<Complex, 3> NeutronScatteringMonitor::compute_polarized_cross_sections(const jams::MultiArray<Vec<Complex,3>, 3>& spectrum, const std::vector<Vec3>& polarizations) {
  const auto num_freqencies = spectrum.size(0);
  const auto num_reciprocal_points = spectrum.size(1);
  const auto num_sites = ::lattice->num_motif_atoms();

  MultiArray<Complex, 3> convolved(polarizations.size(), num_freqencies, num_reciprocal_points);
  convolved.zero();


  for (auto a = 0; a < num_sites; ++a) {
    for (auto b = 0; b < num_sites; ++b) {
      const Vec3 r = lattice->motif_atom(b).fractional_pos - lattice->motif_atom(a).fractional_pos;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        const auto q = paths_[k].hkl;
        const auto Q = unit_vector(paths_[k].xyz);
        const auto prefactor = exp(-kImagTwoPi * dot(q, r)) * form_factors_(k, a) * form_factors_(k, b);
        // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
        for (auto f = 0; f < num_freqencies; ++f) {
          for (auto p = 0; p < polarizations.size(); ++p) {
            const Vec3 P = polarizations[p];

            convolved(p, f, k) += prefactor * kImagOne * dot(P, cross(conj(spectrum(f, k, a)), spectrum(f, k, b)));

            for (auto i : {0, 1, 2}) {
              for (auto j : {0, 1, 2}) {
                convolved(p, f, k) += kImagOne * prefactor * cross(P, Q)[i] * Q[j] * (
                    conj(spectrum(f, k, a)[i]) * spectrum(f, k, b)[j] -
                    conj(spectrum(f, k, a)[j]) * spectrum(f, k, b)[i]);
              }
            }
          }
        }
      }
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

  periodogram_index_counter_ = 0;
  periodogram_counter_ = 0;

  libconfig::Setting &solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  t_sample_ = output_step_freq_ * t_step;
  num_t_samples_ = welch_params_.segment_size;
  double freq_max = 1.0 / (2.0 * t_sample_);
  freq_delta_ = 1.0 / (num_t_samples_ * t_sample_);

  auto kspace_size = lattice->kspace_size();
  auto num_sites = lattice->num_motif_atoms();
  auto num_materials = lattice->num_materials();

  cout << "\n";
  cout << "  number of samples " << num_t_samples_ << "\n";
  cout << "  sampling time (s) " << t_sample_ << "\n";
  cout << "  acquisition time (s) " << t_sample_ * num_t_samples_ << "\n";
  cout << "  frequency resolution (THz) " << freq_delta_ / kTHz << "\n";
  cout << "  maximum frequency (THz) " << freq_max / kTHz << "\n";
  cout << "\n";

  // hkl_path can be a simple list of nodes e.g.
  //     hkl_path = ( [3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0] );
  // or a list of discontinuous paths e.g.
  //    hkl_path = ( ([3.0, 3.0,-3.0], [ 5.0, 5.0,-5.0]),
  //                 ([3.0, 3.0,-2.0], [ 5.0, 5.0,-4.0]));

  auto& cfg_nodes = settings["hkl_path"];

  continuous_path_ranges_.push_back(0);
  if (cfg_nodes[0].isArray()) {
    vector<Vec3> hkl_path_nodes(cfg_nodes.getLength());
    for (auto i = 0; i < cfg_nodes.getLength(); ++i) {
      hkl_path_nodes[i] = Vec3{cfg_nodes[i][0], cfg_nodes[i][1], cfg_nodes[i][2]};
    }

    paths_ = generate_hkl_reciprocal_space_path(hkl_path_nodes, kspace_size);
    continuous_path_ranges_.push_back(continuous_path_ranges_.back() + paths_.size());
  }
  else if (cfg_nodes[0].isList()) {
    for (auto n = 0; n < cfg_nodes.getLength(); ++n) {
      vector<Vec3> hkl_path_nodes(cfg_nodes[n].getLength());
      for (auto i = 0; i < cfg_nodes.getLength(); ++i) {
        hkl_path_nodes[i] = Vec3{cfg_nodes[n][i][0], cfg_nodes[n][i][1], cfg_nodes[n][i][2]};
      }

      auto new_path = generate_hkl_reciprocal_space_path(hkl_path_nodes, kspace_size);

      continuous_path_ranges_.push_back(continuous_path_ranges_.back() + new_path.size());

      paths_.insert(end(paths_), begin(new_path), end(new_path));
    }
  }
  else {
    jams_die("hkl_nodes in neutron-scattering monitor must be a list or a group");
  }

  polarizations_ = {Vec3{0,0,1}, Vec3{0,0,-1}};

  sq_.resize(kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_sites);
  sqw_.resize(welch_params_.segment_size, paths_.size(), num_sites);
  sqw_.zero();

  total_polarized_cross_sections_.resize(polarizations_.size(), welch_params_.segment_size, paths_.size());
  total_polarized_cross_sections_.zero();

  total_unpolarized_cross_section_.resize(welch_params_.segment_size, paths_.size());
  total_unpolarized_cross_section_.zero();

  auto& cfg_form_factors = settings["form_factor"];
  if (cfg_form_factors.getLength() != num_materials) {
    jams_die("In NeutronScatteringMonitor there must be one form factor per material");
  }

  vector<FormFactorG> g_params(num_materials);
  vector<FormFactorJ> j_params(num_materials);

  for (auto i = 0; i < cfg_form_factors.getLength(); ++i) {
    j_params[i].j0 = config_optional<FormFactorParams>(cfg_form_factors[i], "j0", j_params[i].j0);
    j_params[i].j2 = config_optional<FormFactorParams>(cfg_form_factors[i], "j1", j_params[i].j2);
    j_params[i].j4 = config_optional<FormFactorParams>(cfg_form_factors[i], "j2", j_params[i].j4);
    j_params[i].j6 = config_optional<FormFactorParams>(cfg_form_factors[i], "j3", j_params[i].j6);
    g_params[i] = config_required<FormFactorG>(cfg_form_factors[i], "g");
  }

  form_factors_.resize(paths_.size(), num_sites);

  for (auto i = 0; i < paths_.size(); ++i) {
    Vec3 q = paths_[i].xyz;
    for (auto j = 0; j < num_sites; ++j) {
      auto material = lattice->motif_atom(j).material;
      form_factors_(i, j) = form_factor_q(q, g_params[material], j_params[material]);
    }
  }

    /**
     * @warning FFTW_PRESERVE_INPUT is not supported for r2c arrays
     * http://www.fftw.org/doc/One_002dDimensional-DFTs-of-Real-Data.html
     */
  {
    auto s_backup(globals::s);

    fft_plan_to_qspace_ =
        fft_plan_transform_to_reciprocal_space(globals::s.data(), reinterpret_cast<Complex*>(sq_.data()), kspace_size, num_sites);

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
      [kspace_size](const Vec<Complex,3> &a) { return a / sqrt(product(kspace_size)); });


  const int welch_segment_index = periodogram_index_counter_;

  for (auto k = 0; k < paths_.size(); ++k) {
    auto idx = paths_[k].index;
    for (auto a = 0; a < num_sites; ++a) {
      if (paths_[k].conjugate) {
        sqw_(welch_segment_index, k, a) = conj(sq_(idx[0], idx[1], idx[2], a));
      } else {
        sqw_(welch_segment_index, k, a) = sq_(idx[0], idx[1], idx[2], a);
      }
    }
  }

  periodogram_index_counter_++;

  if (periodogram_index_counter_ % welch_params_.segment_size == 0) {
    // calculate periodogram
    auto spectrum = periodogram();
    periodogram_counter_++;

    auto unpolarized_cross_section = compute_unpolarized_cross_section(spectrum);

    std::transform(unpolarized_cross_section.begin(), unpolarized_cross_section.end(),
        total_unpolarized_cross_section_.begin(), total_unpolarized_cross_section_.begin(),
                   [](const Complex&x, const Complex &y) -> Complex { return x + y; });

    auto polarized_cross_sections = compute_polarized_cross_sections(spectrum, polarizations_);

    std::transform(polarized_cross_sections.begin(), polarized_cross_sections.end(),
                   total_polarized_cross_sections_.begin(), total_polarized_cross_sections_.begin(),
                   [](const Complex&x, const Complex &y) -> Complex { return x + y; });

    output_cross_section();

    // shift overlap data to the start of the range
    for (auto i = 0; i < welch_params_.overlap; ++i) {
      for (auto j = 0; j < paths_.size(); ++j) {
        for (auto m = 0; m < num_sites; ++m) {
          sqw_(i, j, m) = sqw_(sqw_.size(0) - welch_params_.overlap + i, j, m);
        }
      }
    }

    // put the pointer to the overlap position
    periodogram_index_counter_ = welch_params_.overlap;
  }
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
  auto plan = fftw_plan_many_dft_r2c(
      rank, transform_size, num_transforms,
      rspace, nembed, stride, dist,
      FFTW_COMPLEX_CAST(kspace), nembed, stride, dist,
      FFTW_MEASURE);

  if (plan == NULL) {
    throw std::runtime_error("FFTW plan failed for NeutronScatteringMonitor");
  }

  return plan;
}

jams::MultiArray<Vec<Complex,3>,3> NeutronScatteringMonitor::periodogram() {
  jams::MultiArray<Vec<Complex,3>,3> spectrum(sqw_.shape());

  const int num_time_samples  = spectrum.size(0);
  const int num_space_samples = spectrum.size(1);
  const int num_sites         = spectrum.size(2);

  int rank              = 1;
  int transform_size[1] = {num_time_samples};
  int num_transforms    = num_space_samples * num_sites * 3;
  int nembed[]          = {num_time_samples};
  int stride            = num_space_samples * num_sites * 3;
  int dist              = 1;

  fftw_plan fft_plan = fftw_plan_many_dft(
      rank,transform_size,num_transforms,
      FFTW_COMPLEX_CAST(spectrum.data()),nembed,stride,dist,
      FFTW_COMPLEX_CAST(spectrum.data()),nembed,stride,dist,
      FFTW_BACKWARD, FFTW_ESTIMATE);

  spectrum = sqw_;

  // window the data and normalize
  for (auto i = 0; i < num_time_samples; ++i) {
    for (auto j = 0; j < num_space_samples; ++j) {
      for (auto m = 0; m < num_sites; ++m) {
        spectrum(i,j,m) *= fft_window_default(i, num_time_samples) / double(num_time_samples);
      }
    }
  }

  fftw_execute(fft_plan);
  fftw_destroy_plan(fft_plan);

  return spectrum;
}

void NeutronScatteringMonitor::output_cross_section() {
  const auto time_points = total_unpolarized_cross_section_.size(0);

  for (auto n = 0; n < continuous_path_ranges_.size()-1; ++n) {

    auto path_begin_idx = continuous_path_ranges_[n];
    auto path_end_idx = continuous_path_ranges_[n+1];

    std::ofstream sqwfile(seedname + "_neutron_scattering_path_" + to_string(n) + ".tsv");

    sqwfile << "index\t";
    sqwfile << "h\t";
    sqwfile << "k\t";
    sqwfile << "l\t";
    sqwfile << "qx\t";
    sqwfile << "qy\t";
    sqwfile << "qz\t";
    sqwfile << "freq_THz\t";
    sqwfile << "energy_meV\t";
    sqwfile << "sigma_unpol_re\t";
    sqwfile << "sigma_unpol_im\t";
    for (auto k = 0; k < total_polarized_cross_sections_.size(0); ++k) {
      sqwfile << "sigma_pol" << to_string(k) << "_re\t";
      sqwfile << "sigma_pol" << to_string(k) << "_im\t";
    }
    sqwfile << "\n";


    auto format_int = [](ostream &os) -> ostream & {
        return os << std::fixed;
    };

    auto format_fix = [](ostream &os) -> ostream & {
        return os << std::setprecision(8) << std::setw(12) << std::fixed;
    };

    auto format_sci = [](ostream &os) -> ostream & {
        return os << std::setprecision(8) << std::setw(12) << std::scientific;
    };

    // sample time is here because the fourier transform in time is not an integral
    // but a discrete sum
    double si_prefactor = t_sample_ * (1.0 / (kTwoPi * kHBar))
                       * pow2((0.5 * kNeutronGFactor * pow2(kElementaryCharge)) / (kElectronMass * pow2(kSpeedOfLight)));

    double barns_unitcell = si_prefactor / (1e-28 * lattice->num_cells());

    // cross section output units are Barns Steradian^-1 Joules^-1 unitcell^-1
    for (auto i = 0; i < (time_points / 2) + 1; ++i) {
      for (auto j = path_begin_idx; j < path_end_idx; ++j) {
        sqwfile << format_int << j << "\t";
        sqwfile << format_fix << paths_[j].hkl << "\t";
        sqwfile << format_fix << paths_[j].xyz << "\t";
        sqwfile << format_fix << (i * freq_delta_ / 1e12) << "\t"; // THz
        sqwfile << format_fix << (i * freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
        sqwfile << format_sci << barns_unitcell * total_unpolarized_cross_section_(i, j).real() / double(periodogram_counter_) << "\t";
        sqwfile << format_sci << barns_unitcell * total_unpolarized_cross_section_(i, j).imag() / double(periodogram_counter_) << "\t";
        for (auto k = 0; k < total_polarized_cross_sections_.size(0); ++k) {
          sqwfile << format_sci << barns_unitcell * total_polarized_cross_sections_(k, i, j).real() / double(periodogram_counter_) << "\t";
          sqwfile << format_sci << barns_unitcell * total_polarized_cross_sections_(k, i, j).imag() / double(periodogram_counter_) << "\t";
        }
        sqwfile << "\n";
      }
      sqwfile << std::endl;
    }

    sqwfile.close();
  }
}

NeutronScatteringMonitor::~NeutronScatteringMonitor() {
  if (fft_plan_to_qspace_) {
    fftw_destroy_plan(fft_plan_to_qspace_);
    fft_plan_to_qspace_ = nullptr;
  }
}
