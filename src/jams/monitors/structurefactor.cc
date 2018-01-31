// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <fftw3.h>

#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/fft.h"
#include "structurefactor.h"

#include "jblib/containers/array.h"

using namespace std;

class Solver;

// We can't guarenttee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

StructureFactorMonitor::StructureFactorMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  settings.lookupValue("output_sublattice", output_sublattice_enabled_);

  time_point_counter_ = 0;

  // create transform arrays for example to apply a Holstein Primakoff transform
  spin_transformations.resize(num_spins);
  transformed_spins.resize(num_spins, 3);

  for (int i = 0; i < num_spins; ++i) {
    spin_transformations[i] = lattice->material(lattice->atom_material_id(i)).transform;
  }

  libconfig::Setting& solver_settings = ::config->lookup("solver");

  double t_step = solver_settings["t_step"];
  double t_run = solver_settings["t_max"];

  double t_sample = output_step_freq_*t_step;
  int    num_samples = int(t_run/t_sample);
  double freq_max    = 1.0/(2.0*t_sample);
         freq_delta  = 1.0/(num_samples*t_sample);

  cout << "\n";
  cout << "  number of samples " << num_samples << "\n";
  cout << "  sampling time (s) " << t_sample << "\n";
  cout << "  acquisition time (s) " << t_sample * num_samples << "\n";
  cout << "  frequency resolution (THz) " << freq_delta/kTHz << "\n";
  cout << "  maximum frequency (THz) " << freq_max/kTHz << "\n";
  cout << "\n";

  // ------------------------------------------------------------------
  // the spin array is a flat 2D array, but is constructed in the lattice
  // class in the order:
  // [Nx, Ny, Nz, M], [Sx, Sy, Sz]
  // where Nx, Ny, Nz are the supercell positions and M is the motif position
  // We can use that to reinterpet the output from the fft as a 5D array
  // ------------------------------------------------------------------
  s_kspace.resize(lattice->kspace_size()[0], lattice->kspace_size()[1], lattice->kspace_size()[2] / 2 + 1, lattice->motif_size(), 3);

  fft_plan_s_rspace_to_kspace = fft_plan_rspace_to_kspace(transformed_spins.data(), s_kspace.data(), lattice->kspace_size(), lattice->motif_size());

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

  auto b_to_k_matrix = lattice->get_unitcell().inverse_matrix();
  auto k_to_b_matrix = lattice->get_unitcell().matrix();

  for (int n = 0, nend = cfg_nodes.getLength(); n < nend; ++n) {

    // transform into reciprocal lattice vectors
    Vec3 cfg_vec = { double(cfg_nodes[n][0]),
                     double(cfg_nodes[n][1]),
                     double(cfg_nodes[n][2]) };

    if (debug_is_enabled()) {
      cout << "cfg_vec: " << cfg_vec << "\n";
    }

    cfg_vec = scale(cfg_vec, ::lattice->kspace_size());

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
      if (int(b_uvw_nodes[n][i]) > ::lattice->kspace_size()[i]) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(b_uvw_nodes[n][0]), int(b_uvw_nodes[n][1]), int(b_uvw_nodes[n][2]));
      }
      if (int(b_uvw_nodes[n+1][i]) > ::lattice->kspace_size()[i]) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(b_uvw_nodes[n+1][0]), int(b_uvw_nodes[n+1][1]), int(b_uvw_nodes[n+1][2]));
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

      bz_lengths.push_back(abs(bz_line_element));

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


  sqw_x.resize(::lattice->motif_size(), num_samples, b_uvw_points.size());
  sqw_y.resize(::lattice->motif_size(), num_samples, b_uvw_points.size());
  sqw_z.resize(::lattice->motif_size(), num_samples, b_uvw_points.size());
}

void StructureFactorMonitor::update(Solver * solver) {
  using std::complex;
  using namespace globals;

  fft_space();
  store_bz_path_data();

  time_point_counter_++;
}

void StructureFactorMonitor::fft_time() {

  const int time_points = sqw_x.size(1);
  const int space_points = sqw_x.size(2);
  const double norm = 1.0/sqrt(time_points);

  jblib::Array<fftw_complex,2> fft_sqw_x(time_points, space_points);
  jblib::Array<fftw_complex,2> fft_sqw_y(time_points, space_points);
  jblib::Array<fftw_complex,2> fft_sqw_z(time_points, space_points);

  jblib::Array<fftw_complex,2> total_sqw_x(time_points, space_points);
  jblib::Array<fftw_complex,2> total_sqw_y(time_points, space_points);
  jblib::Array<fftw_complex,2> total_sqw_z(time_points, space_points);

  jblib::Array<double,2> total_mag_sqw_x(time_points, space_points);
  jblib::Array<double,2> total_mag_sqw_y(time_points, space_points);
  jblib::Array<double,2> total_mag_sqw_z(time_points, space_points);

  for (int i = 0; i < time_points; ++i) {
    for (int j = 0; j < space_points; ++j) {
      total_sqw_x(i, j)[0] = 0.0; total_sqw_x(i, j)[1] = 0.0;
      total_sqw_y(i, j)[0] = 0.0; total_sqw_y(i, j)[1] = 0.0;
      total_sqw_z(i, j)[0] = 0.0; total_sqw_z(i, j)[1] = 0.0;
      total_mag_sqw_x(i, j) = 0.0;
      total_mag_sqw_y(i, j) = 0.0;
      total_mag_sqw_z(i, j) = 0.0;
    }
  }

  int rank       = 1;
  int sizeN[]   = {time_points};
  int howmany    = space_points;
  int inembed[] = {time_points}; int onembed[] = {time_points};
  int istride    = space_points; int ostride    = space_points;
  int idist      = 1;            int odist      = 1;

  fftw_plan fft_plan_time_x = fftw_plan_many_dft(rank,sizeN,howmany,fft_sqw_x.data(),inembed,istride,idist,fft_sqw_x.data(),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_y = fftw_plan_many_dft(rank,sizeN,howmany,fft_sqw_y.data(),inembed,istride,idist,fft_sqw_y.data(),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_z = fftw_plan_many_dft(rank,sizeN,howmany,fft_sqw_z.data(),inembed,istride,idist,fft_sqw_z.data(),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);

  for (int unit_cell_atom = 0; unit_cell_atom < ::lattice->motif_size(); ++unit_cell_atom) {
    for (int i = 0; i < time_points; ++i) {
      for (int j = 0; j < space_points; ++j) {
        fft_sqw_x(i,j)[0] = sqw_x(unit_cell_atom, i, j).real()*fft_window_default(i, time_points);
        fft_sqw_x(i,j)[1] = sqw_x(unit_cell_atom, i, j).imag()*fft_window_default(i, time_points);

        fft_sqw_y(i,j)[0] = sqw_y(unit_cell_atom, i, j).real()*fft_window_default(i, time_points);
        fft_sqw_y(i,j)[1] = sqw_y(unit_cell_atom, i, j).imag()*fft_window_default(i, time_points);

        fft_sqw_z(i,j)[0] = sqw_z(unit_cell_atom, i, j).real()*fft_window_default(i, time_points);
        fft_sqw_z(i,j)[1] = sqw_z(unit_cell_atom, i, j).imag()*fft_window_default(i, time_points);
      }
    }

    fftw_execute(fft_plan_time_x);
    fftw_execute(fft_plan_time_y);
    fftw_execute(fft_plan_time_z);

    // output DSF for each position in the unit cell

    if (output_sublattice_enabled_) {
      std::string unit_cell_sqw_filename = seedname + "_sqw_" + std::to_string(unit_cell_atom) + ".tsv";
      std::ofstream unit_cell_sqw_file(unit_cell_sqw_filename.c_str());

      unit_cell_sqw_file << "# k_index   |\t";
      unit_cell_sqw_file << " total      |\t";
      unit_cell_sqw_file << " u          |\t";
      unit_cell_sqw_file << " v          |\t";
      unit_cell_sqw_file << " w          |\t";
      unit_cell_sqw_file << " freq (THz) |\t";
      unit_cell_sqw_file << "abs(Sx(q,w))|\t";
      unit_cell_sqw_file <<  "Re(Sx(q,w))|\t";
      unit_cell_sqw_file << "Im(Sx(q,w)) |\t";
      unit_cell_sqw_file << "abs(Sy(q,w))|\t";
      unit_cell_sqw_file << "Re(Sy(q,w)) |\t";
      unit_cell_sqw_file << "Im(Sy(q,w)) |\t";
      unit_cell_sqw_file << "abs(Sz(q,w))|\t";
      unit_cell_sqw_file << "Re(Sz(q,w)) |\t";
      unit_cell_sqw_file << "Im(Sz(q,w))\n";

      for (int i = 0; i < (time_points / 2) + 1; ++i) {
        double total_length = 0.0;
        for (int j = 0; j < space_points; ++j) {
          unit_cell_sqw_file << std::setw(5) << std::fixed << j << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << total_length << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][0] / double(::lattice->kspace_size()[0])  << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][1] / double(::lattice->kspace_size()[1])  << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][2] / double(::lattice->kspace_size()[2])  << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << i * freq_delta / 1e12 << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << norm * fft_sqw_x(i, j)[0] << "\t" << norm * fft_sqw_x(i, j)[1] << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << norm * fft_sqw_y(i, j)[0] << "\t" << norm * fft_sqw_y(i, j)[1] << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << norm * fft_sqw_z(i, j)[0] << "\t" << norm * fft_sqw_z(i, j)[1] << "\n";
          total_length += bz_lengths[j];
        }
        unit_cell_sqw_file << std::endl;
      }

      unit_cell_sqw_file.close();
    }

    for (int i = 0; i < time_points; ++i) {
      for (int j = 0; j < space_points; ++j) {
        total_sqw_x(i, j)[0] += norm*fft_sqw_x(i, j)[0];
        total_sqw_x(i, j)[1] += norm*fft_sqw_x(i, j)[1];
        total_mag_sqw_x(i, j) += norm*sqrt(fft_sqw_x(i, j)[0]*fft_sqw_x(i, j)[0] + fft_sqw_x(i, j)[1]*fft_sqw_x(i, j)[1]);

        total_sqw_y(i, j)[0] += norm*fft_sqw_y(i, j)[0];
        total_sqw_y(i, j)[1] += norm*fft_sqw_y(i, j)[1];
        total_mag_sqw_y(i, j) += norm*sqrt(fft_sqw_y(i, j)[0]*fft_sqw_y(i, j)[0] + fft_sqw_y(i, j)[1]*fft_sqw_y(i, j)[1]);

        total_sqw_z(i, j)[0] += norm*fft_sqw_z(i, j)[0];
        total_sqw_z(i, j)[1] += norm*fft_sqw_z(i, j)[1];
        total_mag_sqw_z(i, j) += norm*sqrt(fft_sqw_z(i, j)[0]*fft_sqw_z(i, j)[0] + fft_sqw_z(i, j)[1]*fft_sqw_z(i, j)[1]);

      }
    }
  }

  std::string name = seedname + "_sqw.tsv";
  std::ofstream sqwfile(name.c_str());

  sqwfile << "# k_index   |\t";
  sqwfile << " total      |\t";
  sqwfile << " u          |\t";
  sqwfile << " v          |\t";
  sqwfile << " w          |\t";
  sqwfile << " freq (THz) |\t";
  sqwfile << "abs(Sx(q,w))|\t";
  sqwfile <<  "Re(Sx(q,w))|\t";
  sqwfile << "Im(Sx(q,w)) |\t";
  sqwfile << "abs(Sy(q,w))|\t";
  sqwfile << "Re(Sy(q,w)) |\t";
  sqwfile << "Im(Sy(q,w)) |\t";
  sqwfile << "abs(Sz(q,w))|\t";
  sqwfile << "Re(Sz(q,w)) |\t";
  sqwfile << "Im(Sz(q,w))\n";

  double total_length = 0.0;
  double region_length = 0.0;
  for (int bz_region = 0; bz_region < bz_points_path_count.size() - 1; ++bz_region) {
    for (int i = 0; i < (time_points/2) + 1; ++i) {
      region_length = 0.0;
      for (int j = bz_points_path_count[bz_region]; j < bz_points_path_count[bz_region+1]; ++j) {
        sqwfile << std::setw(5) << std::fixed << j << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << region_length + total_length + 0.5 * bz_lengths[j] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][0] / double(::lattice->kspace_size()[0]) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][1] / double(::lattice->kspace_size()[1]) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << b_uvw_points[j][2] / double(::lattice->kspace_size()[2]) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << i*freq_delta / 1e12 << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_mag_sqw_x(i,j) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_x(i,j)[0] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_x(i,j)[1] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_mag_sqw_y(i,j) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_y(i,j)[0] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_y(i,j)[1] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_mag_sqw_z(i,j) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_z(i,j)[0] << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << total_sqw_z(i,j)[1] << "\n";
        region_length += bz_lengths[j];
      }
      sqwfile << std::endl;
    }
    sqwfile << std::endl;
    total_length += region_length;
  }

  sqwfile.close();
}

StructureFactorMonitor::~StructureFactorMonitor() {
  fft_time();
  if (fft_plan_s_rspace_to_kspace) {
    fftw_free(fft_plan_s_rspace_to_kspace);
  }
}

void StructureFactorMonitor::fft_space() {
  assert(fft_plan_s_rspace_to_kspace != nullptr);
  assert(s_kspace.is_allocated());

  transformed_spins.zero();
  for (auto n = 0; n < globals::num_spins; ++n) {
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        transformed_spins(n, i) += spin_transformations[n][i][j] * globals::s(n, j);
      }
    }
  }

  fftw_execute(fft_plan_s_rspace_to_kspace);

  const double norm = 1.0 / sqrt(product(lattice->kspace_size()));
  for (auto i = 0; i < s_kspace.elements(); ++i) {
    s_kspace[i] *= norm;
  }

  apply_kspace_phase_factors(s_kspace);
}

void StructureFactorMonitor::store_bz_path_data() {
  Vec3i size = lattice->kspace_size();

  for (auto m = 0; m < ::lattice->motif_size(); ++m) {
    for (auto i = 0; i < b_uvw_points.size(); ++i) {
      auto uvw = b_uvw_points[i];
      bool negative = uvw[2] < 0;
      uvw[2] = std::abs(uvw[2]);

      uvw[0] = (size[0] + uvw[0]) % size[0];
      uvw[1] = (size[1] + uvw[1]) % size[1];

      if ( even(uvw[2]/ (size[2]/2 + 1)) ) {
        uvw[2] = ((size[2]/2 + 1) + uvw[2]) % (size[2]/2 + 1);
      } else {
        uvw[2] = (size[2]/2 - 1) + ((size[2]/2 + 1) - uvw[2]) % (size[2]/2 + 1);
      }


      assert(uvw[0] >= 0 && uvw[0] < s_kspace.size(0));
      assert(uvw[1] >= 0 && uvw[1] < s_kspace.size(1));
      assert(uvw[2] >= 0 && uvw[2] < s_kspace.size(2));

//      cout << b_uvw_points[i] << " | " << uvw << "\n";

      if (negative) {
        sqw_x(m, time_point_counter_, i) = conj(s_kspace(uvw[0], uvw[1], uvw[2], m, 0));
        sqw_y(m, time_point_counter_, i) = conj(s_kspace(uvw[0], uvw[1], uvw[2], m, 1));
        sqw_z(m, time_point_counter_, i) = conj(s_kspace(uvw[0], uvw[1], uvw[2], m, 2));
      } else {
        sqw_x(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 0);
        sqw_y(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 1);
        sqw_z(m, time_point_counter_, i) = s_kspace(uvw[0], uvw[1], uvw[2], m, 2);
      }
    }
  }
}