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
#include "spectrum_fourier.h"
#include "jams/helpers/output.h"

using namespace std;

class Solver;

// We can't guarantee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

namespace {
    ostream& float_format(ostream& out)
    {
      return out << std::setprecision(8) << std::setw(12) << std::fixed;
    }
}

SpectrumFourierMonitor::SpectrumFourierMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  settings.lookupValue("output_sublattice", output_sublattice_enabled_);

  time_point_counter_ = 0;

  // create transform arrays for example to apply a Holstein Primakoff transform
  spin_transformations.resize(globals::num_spins);
  transformed_spins.resize(globals::num_spins, 3);

  for (int i = 0; i < globals::num_spins; ++i) {
    spin_transformations[i] = globals::lattice->material(globals::lattice->atom_material_id(i)).transform;
  }

  libconfig::Setting& solver_settings = globals::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_*t_step;
  int    num_samples = ceil(t_run/t_sample);
  double freq_max    = 1.0/(2.0*t_sample);
         freq_delta  = 1.0/(num_samples*t_sample);

  cout << "\n";
  cout << "  number of samples " << num_samples << "\n";
  cout << "  sampling time (s) " << t_sample << "\n";
  cout << "  acquisition time (s) " << t_sample * num_samples << "\n";
  cout << "  frequency resolution (THz) " << freq_delta << "\n";
  cout << "  maximum frequency (THz) " << freq_max << "\n";
  cout << "\n";

  // ------------------------------------------------------------------
  // the spin array is a flat 2D array, but is constructed in the lattice
  // class in the order:
  // [Nx, Ny, Nz, M], [Sx, Sy, Sz]
  // where Nx, Ny, Nz are the supercell positions and M is the motif position
  // We can use that to reinterpet the output from the fft as a 5D array
  // ------------------------------------------------------------------
  s_kspace.resize(globals::lattice->kspace_size()[0], globals::lattice->kspace_size()[1], globals::lattice->kspace_size()[2],
                  globals::lattice->num_motif_atoms(), 3);

  fft_plan_s_rspace_to_kspace = fft_plan_rspace_to_kspace(transformed_spins.data(), s_kspace.data(), globals::lattice->kspace_size(),
                                                          globals::lattice->num_motif_atoms());

  // ------------------------------------------------------------------
  // construct Brillouin zone sample points from the nodes specified
  // in the config file
  // ------------------------------------------------------------------

  // TODO: check that the specified points lie within the BZ

  // loop over node points - the last point is not included in the loop
  // because we move along lines of x_n -> x_n+1
  libconfig::Setting &cfg_nodes = settings["brillouin_zone"];

  int bz_point_counter = 0;

  // read the bz-points from the config

  auto b_to_k_matrix = globals::lattice->get_unitcell().inverse_matrix();
  auto k_to_b_matrix = globals::lattice->get_unitcell().matrix();

  for (int n = 0, nend = cfg_nodes.getLength(); n < nend; ++n) {

    // transform into reciprocal lattice vectors
    Vec3 cfg_vec = { double(cfg_nodes[n][0]),
                     double(cfg_nodes[n][1]),
                     double(cfg_nodes[n][2]) };

    if (debug_is_enabled()) {
      cout << "cfg_vec: " << cfg_vec << "\n";
    }

    cfg_vec = hadamard_product(cfg_vec, globals::lattice->kspace_size());

    auto bz_vec = cfg_vec;

    if (verbose_is_enabled()) {
      cout << "  bz node: ";
      cout << "int[ ";
      for (auto i = 0; i < 3; ++i) {
        cout << int(bz_vec[i]) << " ";
      }
      cout << "] float [ ";
      for (auto i = 0; i < 3; ++i) {
        cout << bz_vec[i] << " ";
      }
      cout << "]\n";
    }

    b_uvw_nodes.push_back({int(bz_vec[0]), int(bz_vec[1]), int(bz_vec[2])});
  }

  cout << "\n";

  bz_points_path_count.push_back(0);
  for (int n = 0, nend = b_uvw_nodes.size()-1; n < nend; ++n) {
    Vec3i bz_point, bz_line, bz_line_element;


    // validate the nodes
    for (int i = 0; i < 3; ++i) {
      if (int(b_uvw_nodes[n][i]) > globals::lattice->kspace_size()[i]) {
        jams_die("bz node point [ %4d %4d %4d ] is larger than the kspace", int(b_uvw_nodes[n][0]),
                 int(b_uvw_nodes[n][1]),
                 int(b_uvw_nodes[n][2]));
      }
      if (int(b_uvw_nodes[n+1][i]) > globals::lattice->kspace_size()[i]) {
        jams_die("bz node point [ %4d %4d %4d ] is larger than the kspace", int(b_uvw_nodes[n + 1][0]),
                 int(b_uvw_nodes[n + 1][1]), int(b_uvw_nodes[n + 1][2]));
      }
    }

    // vector between the nodes
    for (int i = 0; i < 3; ++i) {
      bz_line[i] = int(b_uvw_nodes[n+1][i]) - int(b_uvw_nodes[n][i]);
    }

    if(verbose_is_enabled()) {
      cout << "  bz line: [ " << bz_line[0] << " " << bz_line[1] << " " << bz_line[2] << "\n";
    }

    // normalised vector
    for (int i = 0; i < 3; ++i) {
      bz_line[i] != 0 ? bz_line_element[i] = bz_line[i]/abs(bz_line[i]) : bz_line_element[i] = 0;
    }

    // the number of points is the max dimension in line
    const int bz_line_points = 1 + std::max(std::max(std::abs(bz_line[0]), std::abs(bz_line[1])), std::abs(bz_line[2]));

    if (verbose_is_enabled()) {
      cout << "  bz line points  " << bz_line_points << "\n";
    }

    // store the length element between these points
    for (int j = 0; j < bz_line_points; ++j) {
      for (int i = 0; i < 3; ++i) {
        bz_point[i] = int(b_uvw_nodes[n][i]) + j*bz_line_element[i];
      }

      // check if this is a continuous path and drop duplicate points at the join
      if (b_uvw_points.size() > 0) {
        if(bz_point[0] == b_uvw_points.back()[0]
          && bz_point[1] == b_uvw_points.back()[1]
          && bz_point[2] == b_uvw_points.back()[2]) {
          continue;
        }
      }

      bz_lengths.push_back(norm(bz_line_element));

      b_uvw_points.push_back(bz_point);
      if (verbose_is_enabled()) {
        auto uvw = b_uvw_points.back();
        cout << fixed << setprecision(8);
        cout << "  bz point ";
        cout << setw(4) << bz_point_counter << " ";
        cout << setw(8) << bz_lengths.back() << " ";
        cout << uvw << " ";
        cout << b_to_k_matrix * uvw << "\n";
        cout.unsetf(ios_base::floatfield);
      }
      bz_point_counter++;
    }
    bz_points_path_count.push_back(b_uvw_points.size());
  }


  sqw_x.resize(globals::lattice->num_motif_atoms(), num_samples, b_uvw_points.size());
  sqw_y.resize(globals::lattice->num_motif_atoms(), num_samples, b_uvw_points.size());
  sqw_z.resize(globals::lattice->num_motif_atoms(), num_samples, b_uvw_points.size());
}

void SpectrumFourierMonitor::update(Solver * solver) {
  fft_space();
  store_bz_path_data();

  time_point_counter_++;
}

void SpectrumFourierMonitor::fft_time() {

  const auto time_points = sqw_x.size(1);
  const auto space_points = sqw_x.size(2);
  const double norm = 1.0/sqrt(time_points);

  jams::MultiArray<std::complex<double>,2> fft_sqw_x(time_points, space_points);
  jams::MultiArray<std::complex<double>,2> fft_sqw_y(time_points, space_points);
  jams::MultiArray<std::complex<double>,2> fft_sqw_z(time_points, space_points);

  jams::MultiArray<double,2> total_sqw_x(time_points, space_points);
  jams::MultiArray<double,2> total_sqw_y(time_points, space_points);
  jams::MultiArray<double,2> total_sqw_z(time_points, space_points);

  total_sqw_x.zero();
  total_sqw_y.zero();
  total_sqw_z.zero();

  int rank       = 1;
  int sizeN[]   = {static_cast<int>(time_points)};
  int howmany    = static_cast<int>(space_points);
  int inembed[] = {static_cast<int>(time_points)}; int onembed[] = {static_cast<int>(time_points)};
  int istride    = static_cast<int>(space_points); int ostride    = static_cast<int>(space_points);
  int idist      = 1;            int odist      = 1;

  fftw_plan fft_plan_time_x = fftw_plan_many_dft(
      rank,sizeN,howmany,
      FFTW_COMPLEX_CAST(fft_sqw_x.data()),inembed,istride,idist,
      FFTW_COMPLEX_CAST(fft_sqw_x.data()),onembed,ostride,odist,
      FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_y = fftw_plan_many_dft(
      rank,sizeN,howmany,
      FFTW_COMPLEX_CAST(fft_sqw_y.data()),inembed,istride,idist,
      FFTW_COMPLEX_CAST(fft_sqw_y.data()),onembed,ostride,odist,
      FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_z = fftw_plan_many_dft(
      rank,sizeN,howmany,
      FFTW_COMPLEX_CAST(fft_sqw_z.data()),inembed,istride,idist,
      FFTW_COMPLEX_CAST(fft_sqw_z.data()),onembed,ostride,odist,
      FFTW_FORWARD,FFTW_ESTIMATE);

  for (auto unit_cell_atom = 0; unit_cell_atom < ::globals::lattice->num_motif_atoms(); ++unit_cell_atom) {
    for (auto i = 0; i < time_points; ++i) {
      for (auto j = 0; j < space_points; ++j) {
        fft_sqw_x(i,j) = sqw_x(unit_cell_atom, i, j)*fft_window_default(i, time_points);
        fft_sqw_y(i,j) = sqw_y(unit_cell_atom, i, j)*fft_window_default(i, time_points);
        fft_sqw_z(i,j) = sqw_z(unit_cell_atom, i, j)*fft_window_default(i, time_points);
      }
    }

    fftw_execute(fft_plan_time_x);
    fftw_execute(fft_plan_time_y);
    fftw_execute(fft_plan_time_z);

    // output DSF for each position in the unit cell

    if (output_sublattice_enabled_) {
      std::ofstream unit_cell_sqw_file(jams::output::full_path_filename("sqw_" + std::to_string(unit_cell_atom) + ".tsv"));
      unit_cell_sqw_file.width(12);
      unit_cell_sqw_file << "k_index\t";
      unit_cell_sqw_file << "k_total\t";
      unit_cell_sqw_file << "u\t";
      unit_cell_sqw_file << "v\t";
      unit_cell_sqw_file << "w\t";
      unit_cell_sqw_file << "freq\t";
      unit_cell_sqw_file << "re_sx\t";
      unit_cell_sqw_file << "im_sx\t";
      unit_cell_sqw_file << "re_sy\t";
      unit_cell_sqw_file << "im_sy\t";
      unit_cell_sqw_file << "re_sz\t";
      unit_cell_sqw_file << "im_sz\n";

      for (auto i = 0; i < (time_points / 2) + 1; ++i) {
        double total_length = 0.0;
        for (auto j = 0; j < space_points; ++j) {
          unit_cell_sqw_file << std::setw(5) << std::fixed << j << "\t";
          unit_cell_sqw_file << float_format << total_length << "\t";
          unit_cell_sqw_file << float_format << b_uvw_points[j][0] / double(::globals::lattice->kspace_size()[0]) << "\t";
          unit_cell_sqw_file << float_format << b_uvw_points[j][1] / double(::globals::lattice->kspace_size()[1]) << "\t";
          unit_cell_sqw_file << float_format << b_uvw_points[j][2] / double(::globals::lattice->kspace_size()[2]) << "\t";
          unit_cell_sqw_file << float_format << i * freq_delta / 1e12 << "\t";
          unit_cell_sqw_file << float_format << norm * fft_sqw_x(i, j).real() << "\t" << norm * fft_sqw_x(i, j).imag() << "\t";
          unit_cell_sqw_file << float_format << norm * fft_sqw_y(i, j).real() << "\t" << norm * fft_sqw_y(i, j).imag() << "\t";
          unit_cell_sqw_file << float_format << norm * fft_sqw_z(i, j).real() << "\t" << norm * fft_sqw_z(i, j).imag() << "\n";
          total_length += bz_lengths[j];
        }
        unit_cell_sqw_file << std::endl;
      }

      unit_cell_sqw_file.close();
    }

    for (auto i = 0; i < time_points; ++i) {
      for (auto j = 0; j < space_points; ++j) {
        total_sqw_x(i, j) += norm * std::abs(fft_sqw_x(i, j));
        total_sqw_y(i, j) += norm * std::abs(fft_sqw_y(i, j));
        total_sqw_z(i, j) += norm * std::abs(fft_sqw_z(i, j));
      }
    }
  }

  std::ofstream sqwfile(jams::output::full_path_filename("sqw.tsv"));

  sqwfile.width(12);
  sqwfile << "k_index\t";
  sqwfile << "k_total\t";
  sqwfile << "u\t";
  sqwfile << "v\t";
  sqwfile << "w\t";
  sqwfile << "freq\t";
  sqwfile << "abs_sx\t";
  sqwfile << "abs_sy\t";
  sqwfile << "abs_sz\n";

  double total_length = 0.0;
  double region_length = 0.0;
  for (auto bz_region = 0; bz_region < bz_points_path_count.size() - 1; ++bz_region) {
    for (auto i = 0; i < (time_points/2) + 1; ++i) {
      region_length = 0.0;
      for (auto j = bz_points_path_count[bz_region]; j < bz_points_path_count[bz_region+1]; ++j) {
        sqwfile << std::setw(5) << std::fixed << j << "\t";
        sqwfile << float_format << region_length + total_length + 0.5 * bz_lengths[j] << "\t";
        sqwfile << float_format << b_uvw_points[j][0] / double(::globals::lattice->kspace_size()[0]) << "\t";
        sqwfile << float_format << b_uvw_points[j][1] / double(::globals::lattice->kspace_size()[1]) << "\t";
        sqwfile << float_format << b_uvw_points[j][2] / double(::globals::lattice->kspace_size()[2]) << "\t";
        sqwfile << float_format << i*freq_delta / 1e12 << "\t";
        sqwfile << float_format << sqrt(total_sqw_x(i,j)) << "\t";
        sqwfile << float_format << sqrt(total_sqw_y(i,j)) << "\t";
        sqwfile << float_format << sqrt(total_sqw_z(i,j)) << "\n";
        region_length += bz_lengths[j];
      }
      sqwfile << std::endl;
    }
    sqwfile << std::endl;
    total_length += region_length;
  }

  sqwfile.close();

  fftw_destroy_plan(fft_plan_time_x);
  fftw_destroy_plan(fft_plan_time_y);
  fftw_destroy_plan(fft_plan_time_z);
}

SpectrumFourierMonitor::~SpectrumFourierMonitor() {
  if (fft_plan_s_rspace_to_kspace) {
    fftw_destroy_plan(fft_plan_s_rspace_to_kspace);
    fft_plan_s_rspace_to_kspace = nullptr;
  }
  fft_time();
}

void SpectrumFourierMonitor::fft_space() {
  assert(fft_plan_s_rspace_to_kspace != nullptr);
  assert(s_kspace.elements() > 0);

  transformed_spins.zero();

  for (auto n = 0; n < globals::num_spins; ++n) {
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        transformed_spins(n, i) += complex<double>{spin_transformations[n][i][j] * globals::s(n, j), 0.0};
      }
    }
  }

  fftw_execute(fft_plan_s_rspace_to_kspace);

  const double norm = 1.0 / sqrt(product(globals::lattice->kspace_size()));
  std::transform(s_kspace.begin(), s_kspace.end(), s_kspace.begin(),
      [norm](const std::complex<double> &a)->std::complex<double> {return a * norm;});

  apply_kspace_phase_factors(s_kspace);
}

void SpectrumFourierMonitor::store_bz_path_data() {
  Vec3i size = globals::lattice->kspace_size();

  // extra safety in case there is an extra one time point due to floating point maths
  if (time_point_counter_ < sqw_x.size(1)) {
    for (auto m = 0; m < ::globals::lattice->num_motif_atoms(); ++m) {
      for (auto i = 0; i < b_uvw_points.size(); ++i) {
        auto uvw = b_uvw_points[i];

        uvw = (size + uvw) % size;

        assert(uvw[0] >= 0 && uvw[0] < s_kspace.size(0));
        assert(uvw[1] >= 0 && uvw[1] < s_kspace.size(1));
        assert(uvw[2] >= 0 && uvw[2] < s_kspace.size(2));

        assert(m < sqw_x.size(0));
        assert(time_point_counter_ < sqw_x.size(1));
        assert(i < sqw_x.size(2));

        sqw_x(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 0);
        sqw_y(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 1);
        sqw_z(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 2);
      }
    }
  }
}