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
using Complex = std::complex<double>;

class Solver;

// We can't guarantee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

namespace {
    ostream& float_format(ostream& out) {
      return out << std::setprecision(8) << std::setw(12) << std::fixed;
    }

    inline double polarization_factor(const char alpha, const char beta, const Vec3& q) {
      map<char, int> components = {{'x', 0}, {'y', 1}, {'z', 2}};
      const auto k = zero_safe_normalize(q);
      return kronecker_delta(alpha, beta) - k[components[alpha]]*k[components[beta]];
    }

    jams::MultiArray<Complex, 2> partial_cross_section(const char alpha, const char beta, const unsigned site_a, const unsigned site_b, const jams::MultiArray<Complex, 2>& sqw_a, const jams::MultiArray<Complex, 2>& sqw_b, const vector<Vec3>& qpoints) {
      const auto num_freqencies = sqw_a.size(0);
      const auto num_reciprocal_points = sqw_a.size(1);

      jams::MultiArray<Complex, 2> convolved(num_freqencies, num_reciprocal_points);

      // do convolution a[-w] * b[w] == conj(a[w]) * b[w]
      std::transform(sqw_a.begin(), sqw_a.end(), sqw_b.begin(), convolved.begin(),
          [](const Complex &a, const Complex &b) { return conj(a) * b; });

      // we do the phasing here because r_ij needs to change sign for a->b, b->a

      // THIS IS A FRACTIONAL POSITION
      // TODO: does this need to be periodic across the unit cell boundary?
      Vec3 r = lattice->motif_atom(site_b).pos - lattice->motif_atom(site_a).pos;
      const auto r_cart = lattice->fractional_to_cartesian(r);

      for (auto j = 0; j < num_reciprocal_points; ++j) {
        const auto q = qpoints[j];
        const auto q_cart = lattice->get_unitcell().inverse_matrix()*q;
        const auto phase = exp(-kImagTwoPi * dot(q_cart, r_cart));


        const auto factor = polarization_factor(alpha, beta, q_cart);

        for (auto i = 0; i < num_freqencies; ++i) {
          convolved(i, j) *= phase * factor;
        }
      }

      return convolved;
    }

}

vector<Qpoint> generate_kspace_path(const vector<Vec3>& kspace_nodes, const Vec3i& kspace_size) {
  vector<Vec3i> hkl_nodes;
  Vec3 deltaQ = 1.0 / to_double(kspace_size);

  for (auto node : kspace_nodes) {
    hkl_nodes.push_back(to_int(scale(node, kspace_size)));
  }

  vector<Vec3i> supercell_Qpoints;
  vector<Qpoint> qpoints;

  for (auto n = 0; n < hkl_nodes.size()-1; ++n) {
    const auto origin = hkl_nodes[n];
    const auto path   = hkl_nodes[n+1] - origin;

    // normalised direction vector
    const auto coordinate_delta = normalize_components(path);

    // use +1 to include the last point on the path
    const auto num_coordinates = abs_max(path) + 1;

    auto coordinate = origin;
    for (auto i = 0; i < num_coordinates; ++i) {
      // map an arbitrary coordinate into the limited k indicies of the reduced brillouin zone
      // this is in FFTW ordering so negative indicies wrap around
      auto reduced_coordinate = (coordinate % kspace_size + kspace_size) % kspace_size;
      auto hkl = scale(coordinate, deltaQ);
      auto q = lattice->get_unitcell().inverse_matrix() * hkl;
      qpoints.push_back({hkl, q, reduced_coordinate});
      coordinate += coordinate_delta;
    }
  }

  // remove duplicates in the path (vertices)
  qpoints.erase(std::unique(qpoints.begin(), qpoints.end()), qpoints.end());

  return qpoints;
}


NeutronScatteringMonitor::NeutronScatteringMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  time_point_counter_ = 0;

  transformed_spins_.resize(num_spins, 3);

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
      lattice->kspace_size()[0], lattice->kspace_size()[1], lattice->kspace_size()[2],lattice->num_motif_atoms(), 3);

  fft_plan_s_to_reciprocal_space_ =
      fft_plan_rspace_to_kspace(transformed_spins_.data(), s_reciprocal_space_.data(), lattice->kspace_size(), lattice->num_motif_atoms());

  libconfig::Setting &cfg_nodes = settings["hkl_path"];

  for (auto i = 0; i < cfg_nodes.getLength(); ++i) {
    brillouin_zone_nodes_.push_back(
        Vec3{cfg_nodes[i][0], cfg_nodes[i][1], cfg_nodes[i][2]});
  }

  auto qpoint_data = generate_kspace_path(brillouin_zone_nodes_, lattice->kspace_size());

  for (const auto& p : qpoint_data) {
    hkl_indicies_.push_back(p.hkl);
    q_vectors_.push_back(p.q);
    brillouin_zone_mapping_.push_back(p.index);
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

  bool brute_force_fourier_transform = false;

  if (brute_force_fourier_transform) {
    if (time_point_counter_ < sqw_x_.size(1)) {
      for (auto i = 0; i < q_vectors_.size(); ++i) {
        for (auto n = 0; n < num_spins; ++n) {
          const auto m = lattice->atom_motif_position(n);
          const auto coeff = exp(-kImagTwoPi * dot(lattice->atom_cell_position(n), q_vectors_[i]));
          sqw_x_(m, time_point_counter_, i) += s(n, 0) * coeff;
          sqw_y_(m, time_point_counter_, i) += s(n, 1) * coeff;
          sqw_z_(m, time_point_counter_, i) += s(n, 2) * coeff;
        }
      }
    }
  } else {

    transformed_spins_.zero();

    for (auto n = 0; n < globals::num_spins; ++n) {
      for (auto i = 0; i < 3; ++i) {
        transformed_spins_(n, i) = Complex{globals::s(n, i), 0.0};
      }
    }

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
//        sqwfile << float_format << region_length + total_length + 0.5 * bz_lengths[j] << "\t";
      sqwfile << float_format << hkl_indicies_[j] << "\t";
      sqwfile << float_format << q_vectors_[j] << "\t";
      sqwfile << float_format << (i*freq_delta_ / 1e12) << "\t"; // THz
      sqwfile << float_format << (i*freq_delta_ / 1e12) * 4.135668 << "\t"; // meV
      sqwfile << float_format << total_cross_section(i,j).real() << "\t";
      sqwfile << float_format << total_cross_section(i,j).imag() << "\n";
//        region_length += bz_lengths[j];
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
