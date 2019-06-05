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

// We can't guarantee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

namespace {

    template <typename T, std::size_t N>
    inline void force_multiarray_sync(MultiArray<T,N> & x) {
      volatile auto sync_data = x.data();
    }

    ostream& float_format(ostream& out) {
      return out << std::setprecision(8) << std::setw(12) << std::fixed;
    }

    inline double polarization_factor(const char alpha, const char beta, const Vec3& q) {
      map<char, int> components = {{'x', 0}, {'y', 1}, {'z', 2}};
      const auto k = zero_safe_normalize(q);
      return kronecker_delta(alpha, beta) - k[components[alpha]]*k[components[beta]];
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
    MultiArray<Complex, 2> partial_cross_section(const char alpha, const char beta, const unsigned site_a, const unsigned site_b, const MultiArray<Complex, 2>& sqw_a, const MultiArray<Complex, 2>& sqw_b, const vector<Vec3>& hkl_indicies) {

      const auto num_freqencies = sqw_a.size(0);
      const auto num_reciprocal_points = sqw_a.size(1);

      MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);

      // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
      transform(sqw_a.begin(), sqw_a.end(), sqw_b.begin(), convolved.begin(),
          [](const Complex &a, const Complex &b) { return conj(a) * b; });


      Vec3 r_frac = lattice->motif_atom(site_b).pos - lattice->motif_atom(site_a).pos;

      for (auto j = 0; j < num_reciprocal_points; ++j) {
        const auto q = hkl_indicies[j];
        const auto q_cart = lattice->get_unitcell().inverse_matrix()*q;
        const auto phase = exp(-kImagTwoPi * dot(q, r_frac));


        const auto factor = polarization_factor(alpha, beta, q_cart);

        for (auto i = 0; i < num_freqencies; ++i) {
          convolved(i, j) *= phase * factor;
        }
      }

      return convolved;
    }

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
template <std::size_t Dim>
inline pair<Vec<int,Dim>, bool> fftw_remap_index_real_to_complex(Vec<int,Dim> k, const Vec<int,Dim> &N) {
  auto array_index = (k % N + N) % N;

  if (array_index.back() < N.back()/2+1) {
    // return normal index
    return {array_index, false};
  } else {
    // return Hermitian conjugate
    array_index = (-k % N + N) % N;
    return {array_index, true};
  }
}

/**
 * Generate a path between nodes in reciprocal space sampling the kspace discretely.
 *
 * @param hkl_nodes
 * @param reciprocal_space_size
 * @return
 */
vector<Qpoint> generate_hkl_reciprocal_space_path(const vector<Vec3> &hkl_nodes, const Vec3i &reciprocal_space_size) {

  Vec3 deltaQ = 1.0 / to_double(reciprocal_space_size);

  vector<Vec3i> hkl_nodes_scaled;
  for (auto node : hkl_nodes) {
    hkl_nodes_scaled.push_back(to_int(scale(node, reciprocal_space_size)));
  }

  vector<Qpoint> hkl_path;

  for (auto n = 0; n < hkl_nodes_scaled.size()-1; ++n) {

    const auto hkl_origin = hkl_nodes_scaled[n];
    const auto hkl_displacement = hkl_nodes_scaled[n+1] - hkl_origin;

    // normalised direction vector
    const auto hkl_delta = normalize_components(hkl_displacement);

    // use +1 to include the last point on the hkl_displacement
    const auto num_hkl_coordinates = abs_max(hkl_displacement) + 1;

    auto hkl_coordinate = hkl_origin;
    for (auto i = 0; i < num_hkl_coordinates; ++i) {

      // map an arbitrary hkl_coordinate into the limited k indicies of the reduced brillouin zone
      auto hkl_remapped_index = fftw_remap_index_real_to_complex(hkl_coordinate, reciprocal_space_size);

      cout << hkl_coordinate << "\t" << hkl_remapped_index.first << "\t" << hkl_remapped_index.second << endl;

      auto hkl = scale(hkl_coordinate, deltaQ);
      auto q = lattice->get_unitcell().inverse_matrix() * hkl;
      hkl_path.push_back({hkl, q, hkl_remapped_index.first, hkl_remapped_index.second});
      hkl_coordinate += hkl_delta;
    }
  }

  // remove duplicates in the path where start and end indicies are the same at nodes
  hkl_path.erase(std::unique(hkl_path.begin(), hkl_path.end()), hkl_path.end());

  return hkl_path;
}


fftw_plan fft_plan_transform_to_reciprocal_space(double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & motif_size) {
  assert(rspace != nullptr);
  assert(kspace != nullptr);
  assert(sum(kspace_size) > 0);

  int rank            = 3;
  int stride          = 3 * motif_size;
  int dist            = 1;
  int num_transforms  = 3 * motif_size;
  int transform_size[3]  = {kspace_size[0], kspace_size[1], kspace_size[2]};

  int * nembed = nullptr;

  // NOTE: FFTW_PRESERVE_INPUT is not supported for r2c arrays
  // http://www.fftw.org/doc/One_002dDimensional-DFTs-of-Real-Data.html

  return fftw_plan_many_dft_r2c(
      rank,                    // dimensionality
      transform_size, // array of sizes of each dimension
      num_transforms,          // number of transforms
      rspace,        // input: real data
      nembed,                  // number of embedded dimensions
      stride,                  // memory stride between elements of one fft dataset
      dist,                    // memory distance between fft datasets
      reinterpret_cast<fftw_complex*>(kspace),        // output: complex data
      nembed,                  // number of embedded dimensions
      stride,                  // memory stride between elements of one fft dataset
      dist,                    // memory distance between fft datasets
      FFTW_MEASURE);
}

NeutronScatteringMonitor::NeutronScatteringMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  time_point_counter_ = 0;

  libconfig::Setting& solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run  = solver_settings["t_max"];

  double t_sample    = output_step_freq_*t_step;
  int    num_samples = ceil(t_run/t_sample);
  double freq_max    = 1.0/(2.0*t_sample);
         freq_delta_ = 1.0/(num_samples*t_sample);

  cout << "\n";
  cout << "  number of samples " << num_samples << "\n";
  cout << "  sampling time (s) " << t_sample << "\n";
  cout << "  acquisition time (s) " << t_sample * num_samples << "\n";
  cout << "  frequency resolution (THz) " << freq_delta_/kTHz << "\n";
  cout << "  maximum frequency (THz) " << freq_max/kTHz << "\n";
  cout << "\n";

  // ------------------------------------------------------------------
  // the spin array is a flat 2D array, but is constructed in the lattice
  // class in the order:
  // [Nx, Ny, Nz, M], [Sx, Sy, Sz]
  // where Nx, Ny, Nz are the supercell positions and M is the motif position
  // We can use that to reinterpet the output from the fft as a 5D array
  // ------------------------------------------------------------------
  s_reciprocal_space_.resize(
      lattice->kspace_size()[0], lattice->kspace_size()[1], lattice->kspace_size()[2]/2 + 1,lattice->num_motif_atoms(), 3);
  cout << s(0,0) << "\t" << s(0,1) << "\t" << s(0,2) << endl;



  /**
   * @warning FFTW_PRESERVE_INPUT is not supported for r2c arrays
   * http://www.fftw.org/doc/One_002dDimensional-DFTs-of-Real-Data.html
   */
  // backup input data
  MultiArray<double,2> s_backup(num_spins, 3);
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_backup(i, j) = globals::s(i, j);
    }
  }

  fft_plan_s_to_reciprocal_space_ =
      fft_plan_transform_to_reciprocal_space(globals::s.data(), s_reciprocal_space_.data(), lattice->kspace_size(), lattice->num_motif_atoms());

  // restore data
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      globals::s(i, j) = s_backup(i, j);
    }
  }

  cout << s(0,0) << "\t" << s(0,1) << "\t" << s(0,2) << endl;
  libconfig::Setting &cfg_nodes = settings["hkl_path"];

  for (auto i = 0; i < cfg_nodes.getLength(); ++i) {
    brillouin_zone_nodes_.push_back(
        Vec3{cfg_nodes[i][0], cfg_nodes[i][1], cfg_nodes[i][2]});
  }

  auto qpoint_data = generate_hkl_reciprocal_space_path(brillouin_zone_nodes_, lattice->kspace_size());

  for (const auto& p : qpoint_data) {
    hkl_indicies_.push_back(p.hkl);
    q_vectors_.push_back(p.q);
    brillouin_zone_mapping_.push_back(p.index);
    conjugation_.push_back(p.hermitian);
  }

  sqw_x_.resize(::lattice->num_motif_atoms(), num_samples, brillouin_zone_mapping_.size());
  sqw_y_.resize(::lattice->num_motif_atoms(), num_samples, brillouin_zone_mapping_.size());
  sqw_z_.resize(::lattice->num_motif_atoms(), num_samples, brillouin_zone_mapping_.size());

  sqw_x_.zero();
  sqw_y_.zero();
  sqw_z_.zero();
}

void NeutronScatteringMonitor::update(Solver * solver) {
  using namespace globals;
  assert(fft_plan_s_to_reciprocal_space_ != nullptr);
  assert(s_reciprocal_space_.elements() > 0);

  /**
   * @warning fftw_execute doesn't call s.data() so the CPU and GPU memory don't sync
   * we must force them to sync manually
   */
  force_multiarray_sync(globals::s);
  fftw_execute(fft_plan_s_to_reciprocal_space_);

  const double norm = 1.0 / sqrt(product(lattice->kspace_size()));
  std::transform(s_reciprocal_space_.begin(), s_reciprocal_space_.end(), s_reciprocal_space_.begin(),
                 [norm](const Complex &a) { return a * norm; });


  // store data just on the path of interest

  // extra safety in case there is an extra one time point due to floating point maths
  if (time_point_counter_ < sqw_x_.size(1)) {
    for (auto m = 0; m < ::lattice->num_motif_atoms(); ++m) {
      for (auto i = 0; i < brillouin_zone_mapping_.size(); ++i) {
        auto hkl = brillouin_zone_mapping_[i];
        if (conjugation_[i]) {
          sqw_x_(m, time_point_counter_, i) = conj(s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 0));
          sqw_y_(m, time_point_counter_, i) = conj(s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 1));
          sqw_z_(m, time_point_counter_, i) = conj(s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 2));
        } else {
          sqw_x_(m, time_point_counter_, i) = s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 0);
          sqw_y_(m, time_point_counter_, i) = s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 1);
          sqw_z_(m, time_point_counter_, i) = s_reciprocal_space_(hkl[0], hkl[1], hkl[2], m, 2);
        }
      }
    }
  }

  time_point_counter_++;
}

NeutronScatteringMonitor::~NeutronScatteringMonitor() {
  if (fft_plan_s_to_reciprocal_space_) {
    fftw_destroy_plan(fft_plan_s_to_reciprocal_space_);
    fft_plan_s_to_reciprocal_space_ = nullptr;
  }
}

void NeutronScatteringMonitor::post_process() {
  const auto time_points = sqw_x_.size(1);
  const auto space_points = sqw_x_.size(2);

  jams::MultiArray<Complex,2> total_cross_section(time_points, space_points);
  total_cross_section.zero();

  for (auto site_a = 0; site_a < ::lattice->num_motif_atoms(); ++site_a) {
    auto fft_sqw_a_x = fft_time_to_frequency(site_a, sqw_x_);
    auto fft_sqw_a_y = fft_time_to_frequency(site_a, sqw_y_);
    auto fft_sqw_a_z = fft_time_to_frequency(site_a, sqw_z_);

    for (auto site_b = 0; site_b < ::lattice->num_motif_atoms(); ++site_b) {

      auto fft_sqw_b_x = fft_time_to_frequency(site_b, sqw_x_);
      auto fft_sqw_b_y = fft_time_to_frequency(site_b, sqw_y_);
      auto fft_sqw_b_z = fft_time_to_frequency(site_b, sqw_z_);

      // xx
      {
        auto cross_section = partial_cross_section('x', 'x', site_a, site_b, fft_sqw_a_x, fft_sqw_b_x, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // xy
      {
        auto cross_section = partial_cross_section('x', 'y', site_a, site_b, fft_sqw_a_x, fft_sqw_b_y, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // xz
      {
        auto cross_section = partial_cross_section('x', 'z', site_a, site_b, fft_sqw_a_x, fft_sqw_b_z, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // yx
      {
        auto cross_section = partial_cross_section('y', 'x', site_a, site_b, fft_sqw_a_y, fft_sqw_b_x, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // yy
      {
        auto cross_section = partial_cross_section('y', 'y', site_a, site_b, fft_sqw_a_y, fft_sqw_b_y, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // yz
      {
        auto cross_section = partial_cross_section('y', 'z', site_a, site_b, fft_sqw_a_y, fft_sqw_b_z, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // zx
      {
        auto cross_section = partial_cross_section('z', 'x', site_a, site_b, fft_sqw_a_z, fft_sqw_b_x, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // zy
      {
        auto cross_section = partial_cross_section('z', 'y', site_a, site_b, fft_sqw_a_z, fft_sqw_b_y, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

      // zz
      {
        auto cross_section = partial_cross_section('z', 'z', site_a, site_b, fft_sqw_a_z, fft_sqw_b_z, hkl_indicies_);
        std::transform(total_cross_section.begin(), total_cross_section.end(), cross_section.begin(),
                       total_cross_section.begin(),
                       std::plus<Complex>());
      }

    }
  }

  std::ofstream sqwfile(seedname + "_sqw.tsv");

  sqwfile.width(12);

  // TODO: Update header
  sqwfile << "index\t";
  sqwfile << "h\t";
  sqwfile << "k\t";
  sqwfile << "l\t";
  sqwfile << "qx\t";
  sqwfile << "qy\t";
  sqwfile << "qz\t";
  sqwfile << "freq_THz\t";
  sqwfile << "energy_meV\t";
  sqwfile << "abs_sx\t";
  sqwfile << "abs_sy\t";
  sqwfile << "abs_sz\n";

  for (auto i = 0; i < (time_points/2) + 1; ++i) {
    for (auto j = 0; j < hkl_indicies_.size(); ++j) {
      sqwfile << std::setw(5) << std::fixed << j << "\t";
      sqwfile << float_format << hkl_indicies_[j] << "\t";
      sqwfile << float_format << q_vectors_[j] << "\t";
      sqwfile << float_format << (i*freq_delta_ / 1e12) << "\t"; // THz
      sqwfile << float_format << (i*freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
      sqwfile << std::scientific << total_cross_section(i,j).real() << "\t";
      sqwfile << std::scientific << total_cross_section(i,j).imag() << "\n";
    }
    sqwfile << std::endl;
  }

  sqwfile.close();
}

jams::MultiArray<std::complex<double>, 2> NeutronScatteringMonitor::fft_time_to_frequency(unsigned site, const jams::MultiArray<Complex, 3>& s_time) {

  const auto time_points = s_time.size(1);
  const auto space_points = s_time.size(2);

  jams::MultiArray<Complex,2> s_freq(time_points, space_points);

  int rank       = 1;
  int sizeN[]   = {static_cast<int>(time_points)};
  int howmany    = static_cast<int>(space_points);
  int inembed[] = {static_cast<int>(time_points)}; int onembed[] = {static_cast<int>(time_points)};
  int istride    = static_cast<int>(space_points); int ostride    = static_cast<int>(space_points);
  int idist      = 1;            int odist      = 1;

  fftw_plan fft_plan = fftw_plan_many_dft(
      rank,sizeN,howmany,
      FFTWCAST(s_freq.data()),inembed,istride,idist,
      FFTWCAST(s_freq.data()),onembed,ostride,odist,
      FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

  // fourier transform site a in time
  for (auto i = 0; i < time_points; ++i) {
    for (auto j = 0; j < space_points; ++j) {
      // division is to normalize the fourier transform
      s_freq(i, j) = s_time(site, i, j)  * fft_window_default(i, time_points) / double(time_points);
    }
  }

  fftw_execute(fft_plan);

  fftw_destroy_plan(fft_plan);

  return s_freq;
}
