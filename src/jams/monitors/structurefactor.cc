// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <fftw3.h>

#include "jams/core/error.h"
#include "jams/core/output.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/consts.h"
#include "jams/core/field.h"
#include "jams/core/fft.h"
#include "jams/monitors/structurefactor.h"

#include "jblib/containers/array.h"

class Solver;

// We can't guarenttee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

StructureFactorMonitor::StructureFactorMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output->write("\nInitialising Structure Factor monitor...\n");

  settings.lookupValue("output_sublattice", output_sublattice_enabled_);

  time_point_counter_ = 0;

  // create transform arrays for example to apply a Holstein Primakoff transform
  s_transform.resize(num_spins, 3);

  libconfig::Setting& material_settings = ::config->lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      s_transform(i,n) = material_settings[::lattice->atom_material(i)]["transform"][n];
    }
  }

  libconfig::Setting& sim_settings = ::config->lookup("sim");

  double t_step = sim_settings["t_step"];
  double t_run = sim_settings["t_run"];

  double t_sample = output_step_freq_*t_step;
  int    num_samples = int(t_run/t_sample);
  double freq_max    = 1.0/(2.0*t_sample);
         freq_delta  = 1.0/(num_samples*t_sample);

  ::output->write("\n");
  ::output->write("  number of samples:          %d\n", num_samples);
  ::output->write("  sampling time (s):          %e\n", t_sample);
  ::output->write("  acquisition time (s):       %e\n", t_sample * num_samples);
  ::output->write("  frequency resolution (THz): %f\n", freq_delta/kTHz);
  ::output->write("  maximum frequency (THz):    %f\n", freq_max/kTHz);
  ::output->write("\n");

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
  for (int n = 0, nend = cfg_nodes.getLength(); n < nend; ++n) {

    // transform into reciprocal lattice vectors
    bz_cfg_points.push_back(
      {double(cfg_nodes[n][0]),
       double(cfg_nodes[n][1]),
       double(cfg_nodes[n][2])});

    Vec3 bz_vec;

    for (int i = 0; i < 3; ++i) {
      bz_vec = bz_vec + ::lattice->inv_unit_cell_vector(i) * bz_cfg_points.back()[i];
    }


    bz_nodes.push_back({int(bz_vec[0]), int(bz_vec[1]), int(bz_vec[2])});
  }

  bz_points_path_count.push_back(0);
  for (int n = 0, nend = bz_nodes.size()-1; n < nend; ++n) {
    jblib::Vec3<int> bz_point, bz_line, bz_line_element;


    // validate the nodes
    for (int i = 0; i < 3; ++i) {
      if (int(bz_nodes[n][i]) > ::lattice->kspace_size()[i]) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n][0]), int(bz_nodes[n][1]), int(bz_nodes[n][2]));
      }
      if (int(bz_nodes[n+1][i]) > ::lattice->kspace_size()[i]) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n+1][0]), int(bz_nodes[n+1][1]), int(bz_nodes[n+1][2]));
      }
    }

    // vector between the nodes
    for (int i = 0; i < 3; ++i) {
      bz_line[i] = int(bz_nodes[n+1][i]) - int(bz_nodes[n][i]);
    }
    ::output->verbose("  bz line: [ %4d %4d %4d ]\n", bz_line[0], bz_line[1], bz_line[2]);

    // normalised vector
    for (int i = 0; i < 3; ++i) {
      bz_line[i] != 0 ? bz_line_element[i] = bz_line[i]/abs(bz_line[i]) : bz_line_element[i] = 0;
    }

    // the number of points is the max dimension in line
    const int bz_line_points = 1 + std::max(std::max(std::abs(bz_line[0]), std::abs(bz_line[1])), std::abs(bz_line[2]));

    ::output->verbose("  bz line points: %d\n", bz_line_points);

    // store the length element between these points
    for (int j = 0; j < bz_line_points; ++j) {
      for (int i = 0; i < 3; ++i) {
        bz_point[i] = int(bz_nodes[n][i]) + j*bz_line_element[i];
      }

      // check if this is a continuous path and drop duplicate points at the join
      if (bz_points.size() > 0) {
        if(bz_point[0] == bz_points.back()[0]
          && bz_point[1] == bz_points.back()[1]
          && bz_point[2] == bz_points.back()[2]) {
          continue;
        }
      }

      bz_lengths.push_back(abs(bz_line_element));

      bz_points.push_back(bz_point);
      ::output->verbose("  bz point: %6d %6.6f [ %4d %4d %4d ]\n", bz_point_counter, bz_lengths.back(), bz_points.back()[0], bz_points.back()[1], bz_points.back()[2]);
      bz_point_counter++;
    }
    bz_points_path_count.push_back(bz_points.size());
  }


  sqw_x.resize(::lattice->num_unit_cell_positions(), num_samples, bz_points.size());
  sqw_y.resize(::lattice->num_unit_cell_positions(), num_samples, bz_points.size());
  sqw_z.resize(::lattice->num_unit_cell_positions(), num_samples, bz_points.size());

  k0.resize(num_samples);
  kneq0.resize(num_samples);
}

void StructureFactorMonitor::update(Solver * solver) {
  using std::complex;
  using namespace globals;

  jblib::Array<complex<double>, 4> sq_x, sq_y, sq_z;

  jblib::Array<double, 2> s_trans(num_spins, 3);

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      s_trans(i, j) = s(i, j) * s_transform(i, j);
    }
  }

  for (int i = 0; i < num_spins; ++i) {
      s_trans(i, 2) = 1.0 - s_trans(i, 2);
  }

  fft_vector_field(s_trans, sq_x, sq_y, sq_z);

  // add the Sq to the timeseries
  for (int n = 0; n < ::lattice->num_unit_cell_positions(); ++n) {
    for (int i = 0, iend = bz_points.size(); i < iend; ++i) {
      jblib::Vec3<int> q = bz_points[i];
      for (int j = 0; j < 3; ++j) {
        if (q[j] < 0) {
          int nk = ::lattice->kspace_size()[j];
          q[j] = (nk + q[j]);
        }
      }
      sqw_x(n, time_point_counter_, i) = sq_x(q[0], q[1], q[2], n);
      sqw_y(n, time_point_counter_, i) = sq_y(q[0], q[1], q[2], n);
      sqw_z(n, time_point_counter_, i) = sq_z(q[0], q[1], q[2], n);
    }
  }

  jblib::Array<double, 3> nz(sq_z.size(0), sq_z.size(1), sq_z.size(2));

  for (int i = 0; i < nz.size(0); ++i) {
      for (int j = 0; j < nz.size(1); ++j) {
          for (int k = 0; k < nz.size(2); ++k) {
            nz(i, j, k) = norm(sq_z(i, j, k, 0));
          }
      }
  }
  k0(time_point_counter_) = nz(0, 0, 0);
  kneq0(time_point_counter_) = std::accumulate(nz.data(), nz.data()+nz.elements(), 0.0);

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

  for (int unit_cell_atom = 0; unit_cell_atom < ::lattice->num_unit_cell_positions(); ++unit_cell_atom) {
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
      unit_cell_sqw_file << " ky         |\t";
      unit_cell_sqw_file << " kz         |\t";
      unit_cell_sqw_file << " kx         |\t";
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
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][0] / double(::lattice->kspace_size()[0])  << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][1] / double(::lattice->kspace_size()[1])  << "\t";
          unit_cell_sqw_file << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][2] / double(::lattice->kspace_size()[2])  << "\t";
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
  sqwfile << " kx         |\t";
  sqwfile << " ky         |\t";
  sqwfile << " kz         |\t";
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
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][0] / double(::lattice->kspace_size()[0]) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][1] / double(::lattice->kspace_size()[1]) << "\t";
        sqwfile << std::setprecision(8) << std::setw(12) << std::fixed << bz_points[j][2] / double(::lattice->kspace_size()[2]) << "\t";
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
}
