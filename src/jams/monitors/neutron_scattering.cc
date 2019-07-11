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
MultiArray<Complex, 2> NeutronScatteringMonitor::compute_unpolarized_cross_section() {
  const auto num_freqencies = sqw_.size(0);
  const auto num_reciprocal_points = sqw_.size(1);
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
              convolved(f, k) += prefactor * (kronecker_delta(i, j) - Q[i] * Q[j]) * conj(sqw_(f, k, a)[i]) * sqw_(f, k, b)[j];
            }
          }
        }
      }
    }
  }
  return convolved;
}

MultiArray<Complex, 2> NeutronScatteringMonitor::compute_polarized_cross_section(const Vec3& P) {
  const auto num_freqencies = sqw_.size(0);
  const auto num_reciprocal_points = sqw_.size(1);
  const auto num_sites = ::lattice->num_motif_atoms();

  MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);
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

          convolved(f, k) += prefactor * kImagOne * dot(P, cross(conj(sqw_(f, k, a)), sqw_(f, k, b)));

          for (auto i : {0,1,2}) {
            for (auto j : {0,1,2}) {
              convolved(f, k) += kImagOne * prefactor * cross(P,Q)[i] * Q[j] * (
                  conj(sqw_(f, k, a)[i]) * sqw_(f, k, b)[j] - conj(sqw_(f, k, a)[j]) * sqw_(f, k, b)[i] );
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

  time_point_counter_ = 0;

  libconfig::Setting &solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  t_sample_ = output_step_freq_ * t_step;
  num_t_samples_ = ceil(t_run / t_sample_);
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
  sqw_.resize(num_t_samples_, paths_.size(), num_sites);
  sqw_.zero();

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
    auto s_backup = globals::s;

    fft_plan_to_qspace_ =
        fft_plan_transform_to_reciprocal_space(globals::s.data(), &sq_(0,0,0,0)[0], kspace_size, num_sites);

    globals::s = s_backup;
  }
}

void NeutronScatteringMonitor::update(Solver * solver) {
  using namespace globals;
  assert(fft_plan_to_qspace_ != nullptr);
  assert(sq_.elements() > 0);

  // store data just on the path of interest
  // extra safety in case there is an extra one time point due to floating point maths
  if (time_point_counter_ >= sqw_.size(0)) return;

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


  for (auto k = 0; k < paths_.size(); ++k) {
    auto idx = paths_[k].index;
    for (auto a = 0; a < num_sites; ++a) {
      if (paths_[k].conjugate) {
        sqw_(time_point_counter_, k, a) = conj(sq_(idx[0], idx[1], idx[2], a));
      } else {
        sqw_(time_point_counter_, k, a) = sq_(idx[0], idx[1], idx[2], a);
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
      FFTW_COMPLEX_CAST(kspace), nembed, stride, dist,
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
        sqw_(i,j,m) *= fft_window_default(i, num_time_samples) / double(num_time_samples);
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
      FFTW_COMPLEX_CAST(sqw_.data()),nembed,stride,dist,
      FFTW_COMPLEX_CAST(sqw_.data()),nembed,stride,dist,
      FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

  fftw_execute(fft_plan);
  fftw_destroy_plan(fft_plan);
}

void NeutronScatteringMonitor::post_process() {
  const auto time_points = sqw_.size(0);

  fft_to_frequency();

  auto unpolarized_cross_section = compute_unpolarized_cross_section();

  vector<MultiArray<Complex,2>> polarized_cross_sections;
  for (const auto& P : polarizations_) {
    polarized_cross_sections.emplace_back(compute_polarized_cross_section(P));
  }


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
    for (auto k = 0; k < polarized_cross_sections.size(); ++k) {
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
        sqwfile << format_sci << barns_unitcell * unpolarized_cross_section(i, j).real() << "\t";
        sqwfile << format_sci << barns_unitcell * unpolarized_cross_section(i, j).imag() << "\t";
        for (auto k = 0; k < polarized_cross_sections.size(); ++k) {
          sqwfile << format_sci << barns_unitcell * polarized_cross_sections[k](i, j).real() << "\t";
          sqwfile << format_sci << barns_unitcell * polarized_cross_sections[k](i, j).imag() << "\t";
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
