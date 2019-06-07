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

struct FormFactorParams {
    double A, a, B, b, C, c, D; };

struct FormFactorG {
    double g0, g2, g4, g6; };

struct FormFactorJ {
    FormFactorParams j0, j2, j4, j6; };

namespace jams {
    template<>
    inline FormFactorParams config_required(const libconfig::Setting &setting, const std::string &name) {
      return {double(setting[name][0]),
              double(setting[name][1]),
              double(setting[name][2]),
              double(setting[name][3]),
              double(setting[name][4]),
              double(setting[name][5]),
              double(setting[name][6])};
    }

    template<>
    inline FormFactorG config_required(const libconfig::Setting &setting, const std::string &name) {
      return {double(setting[name][0]),
              double(setting[name][1]),
              double(setting[name][2]),
              double(setting[name][3])};
    }
}

double form_factor_jl(const unsigned& l, const double& s, const FormFactorParams& f) {
  if (f.A == 0.0 && f.B == 0.0 && f.C == 0.0 && f.D == 0.0) return 0.0;

  double s2 = s * s;
  double p;
  (l == 0) ? p = 1.0 : p = s2;

  return f.A * p * exp(-f.a * s2) + f.B * p * exp(-f.b * s2) + f.C * p * exp(-f.c * s2) + f.D * p;
}

double form_factor_s(const double& s, const FormFactorG& g, const FormFactorJ& j) {
  double total = 0.0;
  return  0.5 * ( g.g0 * form_factor_jl(0, s, j.j0)
         + g.g2 * form_factor_jl(2, s, j.j2)
         + g.g4 * form_factor_jl(4, s, j.j4)
         + g.g6 * form_factor_jl(6, s, j.j6) );
}

double form_factor_q(const Vec3& q, const FormFactorG& g, const FormFactorJ& j) {
  // crystal tables assume s is in Angstroms^-1 so we convert lattice parameter into Angstroms
  auto s = norm(q) * (1.0/(1e10*lattice->parameter())) / (4.0*kPi);
  return form_factor_s(s, g, j);
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
MultiArray<Complex, 2> NeutronScatteringMonitor::compute_unpolarized_cross_section() {
  const auto num_freqencies = sqw_.size(0);
  const auto num_reciprocal_points = sqw_.size(1);

  MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);
  convolved.zero();

  for (auto site_a = 0; site_a < ::lattice->num_motif_atoms(); ++site_a) {
    for (auto site_b = 0; site_b < ::lattice->num_motif_atoms(); ++site_b) {
      const Vec3 r_frac = lattice->motif_atom(site_b).pos - lattice->motif_atom(site_a).pos;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        const auto q_frac = path_[k].hkl;
        const auto Q = unit_vector(path_[k].xyz);
        const auto phase = exp(-kImagTwoPi * dot(q_frac, r_frac)) * form_factors_(k, site_a) * form_factors_(k, site_b);
        // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
        for (auto f = 0; f < num_freqencies; ++f) {
          // loop xx, xy, ... yz zz
          for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
              convolved(f, k) += (kronecker_delta(i, j) - Q[i] * Q[j])
                                 * phase * conj(sqw_(f, k, site_a, i)) * sqw_(f, k, site_b, j);
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

  MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);
  convolved.zero();

  for (auto site_a = 0; site_a < ::lattice->num_motif_atoms(); ++site_a) {
    for (auto site_b = 0; site_b < ::lattice->num_motif_atoms(); ++site_b) {
      const Vec3 r_frac = lattice->motif_atom(site_b).pos - lattice->motif_atom(site_a).pos;
      for (auto k = 0; k < num_reciprocal_points; ++k) {
        const auto q_frac = path_[k].hkl;
        const auto Q = unit_vector(path_[k].xyz);
        const auto phase = exp(-kImagTwoPi * dot(q_frac, r_frac));
        // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
        for (auto f = 0; f < num_freqencies; ++f) {
          Vec<Complex,3> sxs = {Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}};

          // yz - zy
          sxs[0] += conj(sqw_(f, k, site_a, 1)) * sqw_(f, k, site_b, 2)
                  - conj(sqw_(f, k, site_a, 2)) * sqw_(f, k, site_b, 1);

          // zx - xz
          sxs[1] += conj(sqw_(f, k, site_a, 2)) * sqw_(f, k, site_b, 0)
                  - conj(sqw_(f, k, site_a, 0)) * sqw_(f, k, site_b, 2);

          // xy - yx
          sxs[2] += conj(sqw_(f, k, site_a, 0)) * sqw_(f, k, site_b, 1)
                  - conj(sqw_(f, k, site_a, 1)) * sqw_(f, k, site_b, 0);

          convolved(f, k) += kImagOne * dot(P, phase * sxs);

          for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
              convolved(f, k) += kImagOne * phase * cross(P,Q)[i] * Q[j] * (
                  conj(sqw_(f, k, site_a, i)) * sqw_(f, k, site_b, j)
                  - conj(sqw_(f, k, site_a, j)) * sqw_(f, k, site_b, i)
                  );

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

  double t_sample = output_step_freq_ * t_step;
  int num_samples = ceil(t_run / t_sample);
  double freq_max = 1.0 / (2.0 * t_sample);
  freq_delta_ = 1.0 / (num_samples * t_sample);

  auto kspace_size = lattice->kspace_size();
  auto num_sites = lattice->num_motif_atoms();
  auto num_materials = lattice->num_materials();

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

  polarizations_ = {Vec3{0,0,1}, Vec3{0,0,-1}};

  sq_.resize(kspace_size[0], kspace_size[1], kspace_size[2] / 2 + 1, num_sites, 3);
  sqw_.resize(num_samples, path_.size(), num_sites, 3);
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

  form_factors_.resize(path_.size(), num_sites);

  for (auto i = 0; i < path_.size(); ++i) {
    Vec3 q = path_[i].xyz;
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

  fft_to_frequency();

  auto unpolarized_cross_section = compute_unpolarized_cross_section();

  vector<MultiArray<Complex,2>> polarized_cross_sections;
  for (const auto& P : polarizations_) {
    polarized_cross_sections.emplace_back(compute_polarized_cross_section(P));
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

  // ((gamma/2) * e^2 / m_e c^2) * (1/(2*pi*hbar))
  double prefactor = (1.0 / (kTwoPi * kHBar))
      * (0.5*kNeutronGFactor * pow2(kElementaryCharge))/(kElectronMass * pow2(kSpeedOfLight));

  for (auto i = 0; i < (time_points/2) + 1; ++i) {
    for (auto j = 0; j < path_.size(); ++j) {
      sqwfile << format_int << j << "\t";
      sqwfile << format_fix << path_[j].hkl << "\t";
      sqwfile << format_fix << path_[j].xyz << "\t";
      sqwfile << format_fix << (i*freq_delta_ / 1e12) << "\t"; // THz
      sqwfile << format_fix << (i*freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
      sqwfile << format_sci << prefactor*unpolarized_cross_section(i,j).real() << "\t";
      sqwfile << format_sci << prefactor*unpolarized_cross_section(i,j).imag() << "\t";
      for (auto k = 0; k < polarized_cross_sections.size(); ++k) {
        sqwfile << format_sci << prefactor*polarized_cross_sections[k](i,j).real() << "\t";
        sqwfile << format_sci << prefactor*polarized_cross_sections[k](i,j).imag() << "\t";
      }
      sqwfile << "\n";
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
