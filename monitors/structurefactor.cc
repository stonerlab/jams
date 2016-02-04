// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

#include <fftw3.h>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/consts.h"

#include "monitors/structurefactor.h"

// We can't guarenttee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

StructureFactorMonitor::StructureFactorMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Structure Factor monitor...\n");

  is_equilibration_monitor_ = false;
  time_point_counter_ = 0;

  // plan FFTW routines
  sq_x.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);
  fft_plan_sq_x = fftw_plan_dft_3d(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, sq_x.data(),  sq_x.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  sq_y.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);
  fft_plan_sq_y = fftw_plan_dft_3d(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, sq_y.data(),  sq_y.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  sq_z.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);
  fft_plan_sq_z = fftw_plan_dft_3d(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, sq_z.data(),  sq_z.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  // create transform arrays for example to apply a Holstein Primakoff transform
  s_transform.resize(num_spins, 3);

  libconfig::Setting& material_settings = ::config.lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      s_transform(i,n) = material_settings[::lattice.atom_material(i)]["transform"][n];
    }
  }

  libconfig::Setting& sim_settings = ::config.lookup("sim");

  double t_step = sim_settings["t_step"];
  double t_run = sim_settings["t_run"];

  double    t_sample = output_step_freq_*t_step;
  int    num_samples = int(t_run/t_sample);
  double freq_sample = num_samples / t_run;
  double freq_max    = 1.0/(2.0*t_sample);
         freq_delta  = 1.0 / t_sample;

  ::output.write("\n");
  ::output.write("  number of samples:          %d\n", num_samples);
  ::output.write("  sampling time (s):          %e\n", t_sample);
  ::output.write("  acquisition time (s):       %e\n", t_sample * num_samples);
  ::output.write("  frequency resolution (THz): %f\n", freq_delta/kTHz);
  ::output.write("  maximum frequency (THz):    %f\n", freq_max/kTHz);
  ::output.write("\n");

  // ------------------------------------------------------------------
  // construct Brillouin zone sample points from the nodes specified
  // in the config file
  // ------------------------------------------------------------------

  // TODO: check that the specified points lie within the BZ

  // loop over node points - the last point is not included in the loop
  // because we move along lines of x_n -> x_n+1
  libconfig::Setting &bz_nodes = settings["brillouin_zone"];

  int bz_point_counter = 0;
  for (int n = 0, nend = bz_nodes.getLength()-1; n < nend; ++n) {
    jblib::Vec3<int> bz_point, bz_line, bz_line_element;

    // validate the nodes
    for (int i = 0; i < 3; ++i) {
      if (int(bz_nodes[n][i]) > (lattice.kspace_size()[i]/2 + 1)) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n][0]), int(bz_nodes[n][1]), int(bz_nodes[n][2]));
      }
      if (int(bz_nodes[n+1][i]) > (lattice.kspace_size()[i]/2 + 1)) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n+1][0]), int(bz_nodes[n+1][1]), int(bz_nodes[n+1][2]));
      }
    }

    // vector between the nodes
    for (int i = 0; i < 3; ++i) {
      bz_line[i] = int(bz_nodes[n+1][i]) - int(bz_nodes[n][i]);
    }
    ::output.verbose("  bz line: [ %4d %4d %4d ]\n", bz_line.x, bz_line.y, bz_line.z);

    // normalised vector
    for (int i = 0; i < 3; ++i) {
      bz_line[i] != 0 ? bz_line_element[i] = bz_line[i]/abs(bz_line[i]) : bz_line_element[i] = 0;
    }

    // the number of points is the max dimension in line
    const int bz_line_points = abs(*std::max_element(bz_line.begin(), bz_line.end(), [] (int a, int b) { return (abs(a) < abs(b)); }));
    ::output.verbose("  bz line points: %d\n", bz_line_points);

    // store the length element between these points
    for (int j = 0; j < bz_line_points; ++j) {
      bz_lengths.push_back(abs(bz_line_element));
      for (int i = 0; i < 3; ++i) {
        bz_point[i] = int(bz_nodes[n][i]) + j*bz_line_element[i];
      }
      bz_points.push_back(bz_point);
      ::output.write("  bz point: %6d %6.6f [ %4d %4d %4d ]\n", bz_point_counter, bz_lengths.back(), bz_points.back().x, bz_points.back().y, bz_points.back().z);
      bz_point_counter++;
    }
  }

  sqw_x.resize(lattice.num_unit_cell_positions(), num_samples, bz_points.size());
  sqw_y.resize(lattice.num_unit_cell_positions(), num_samples, bz_points.size());
  sqw_z.resize(lattice.num_unit_cell_positions(), num_samples, bz_points.size());


  chi_xy.resize(num_samples);
  chi_yx.resize(num_samples);

}

void StructureFactorMonitor::update(Solver * solver) {
  using std::complex;
  using namespace globals;

  complex<double> two_pi_i_dr;
  complex<double> exp_phase_0;
  jblib::Array<complex<double>, 1> exp_phase_x(lattice.kspace_size().x);
  jblib::Array<complex<double>, 1> exp_phase_y(lattice.kspace_size().y);
  jblib::Array<complex<double>, 1> exp_phase_z(lattice.kspace_size().z);

  chi_xy[time_point_counter_][0] = 0.0;
  chi_xy[time_point_counter_][1] = 0.0;

  for (int unit_cell_atom = 0; unit_cell_atom < lattice.num_unit_cell_positions(); ++unit_cell_atom) {
    // zero the sq arrays
    for (int i = 0; i < sq_x.elements(); ++i) {
      sq_x[i][0] = 0.0; sq_x[i][1] = 0.0;
      sq_y[i][0] = 0.0; sq_y[i][1] = 0.0;
      sq_z[i][0] = 0.0; sq_z[i][1] = 0.0;
    }

    // remap spin data into kspace array
    for (int i = 0; i < num_spins; ++i) {
      if ((i+unit_cell_atom)%lattice.num_unit_cell_positions() == 0) {
        jblib::Vec3<int> r = lattice.super_cell_pos(i);
        sq_x(r.x, r.y, r.z)[0] = s(i,0)*s_transform(i,0);  sq_x(r.x, r.y, r.z)[1] = 0.0;
        sq_y(r.x, r.y, r.z)[0] = s(i,1)*s_transform(i,1);  sq_y(r.x, r.y, r.z)[1] = 0.0;
        sq_z(r.x, r.y, r.z)[0] = s(i,2)*s_transform(i,2);  sq_z(r.x, r.y, r.z)[1] = 0.0;
      }
    }

    // perform in place FFT
    fftw_execute(fft_plan_sq_x);
    fftw_execute(fft_plan_sq_y);
    fftw_execute(fft_plan_sq_z);

    const double norm = 1.0/sqrt(product(lattice.kspace_size()));


    // super speed hack from CASTEP ewald.f90 for generating all of the phase factors on
    // the fly without calling lots of exp()
    two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(unit_cell_atom).x;
    exp_phase_0 = exp(two_pi_i_dr);
    exp_phase_x(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().x - 1)));
    for (int i = 1; i < lattice.kspace_size().x; ++i) {
      exp_phase_x(i) = exp_phase_x(i-1)*exp_phase_0;
    }

    two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(unit_cell_atom).y;
    exp_phase_0 = exp(two_pi_i_dr);
    exp_phase_y(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().y - 1)));
    for (int i = 1; i < lattice.kspace_size().y; ++i) {
      exp_phase_y(i) = exp_phase_y(i-1)*exp_phase_0;
    }

    two_pi_i_dr = kImagTwoPi* lattice.unit_cell_position_cart(unit_cell_atom).z;
    exp_phase_0 = exp(two_pi_i_dr);
    exp_phase_z(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().z - 1)));
    for (int i = 1; i < lattice.kspace_size().z; ++i) {
      exp_phase_z(i) = exp_phase_z(i-1)*exp_phase_0;
    }





    for (int i = 0; i < lattice.kspace_size().x; ++i) {
      for (int j = 0; j < lattice.kspace_size().y; ++j) {
        for (int k = 0; k < lattice.kspace_size().z; ++k) {

          complex<double> phased_sq_x(sq_x(i, j, k)[0], sq_x(i, j, k)[1]);
          complex<double> phased_sq_y(sq_y(i, j, k)[0], sq_y(i, j, k)[1]);
          complex<double> phased_sq_z(sq_z(i, j, k)[0], sq_z(i, j, k)[1]);

          phased_sq_x = norm*phased_sq_x*exp_phase_x(i)*exp_phase_y(j)*exp_phase_z(k);
          sq_x(i, j, k)[0] = phased_sq_x.real(); sq_x(i, j, k)[1] = phased_sq_x.imag();

          phased_sq_y = norm*phased_sq_y*exp_phase_x(i)*exp_phase_y(j)*exp_phase_z(k);
          sq_y(i, j, k)[0] = phased_sq_y.real(); sq_y(i, j, k)[1] = phased_sq_y.imag();

          phased_sq_z = norm*phased_sq_z*exp_phase_x(i)*exp_phase_y(j)*exp_phase_z(k);
          sq_z(i, j, k)[0] = phased_sq_z.real(); sq_z(i, j, k)[1] = phased_sq_z.imag();
        }
      }
    }

    chi_xy[time_point_counter_][0] = ((s(0, 0)*ds_dt(0, 1) - s(0, 1)*ds_dt(0, 0))); //(sq_x(0,0,0)[0]*sq_y(0,0,0)[0]) + (sq_x(0,0,0)[1]*sq_y(0,0,0)[1]);
    chi_xy[time_point_counter_][1] = 0.0; //(sq_x(0,0,0)[1]*sq_y(0,0,0)[0]) - (sq_x(0,0,0)[0]*sq_y(0,0,0)[1]);


    chi_yx[time_point_counter_][0] = (sq_x(0,0,0)[0]*sq_y(0,0,0)[0]) + (sq_x(0,0,0)[1]*sq_y(0,0,0)[1]);
    chi_yx[time_point_counter_][1] = (sq_x(0,0,0)[0]*sq_y(0,0,0)[1]) - (sq_x(0,0,0)[1]*sq_y(0,0,0)[0]);

    // add the Sq to the timeseries
    for (int i = 0, iend = bz_points.size(); i < iend; ++i) {
      jblib::Vec3<int> q = bz_points[i];
      sqw_x(unit_cell_atom, time_point_counter_, i) = norm*complex<double>(sq_x(q.x, q.y, q.z)[0],sq_x(q.x, q.y, q.z)[1]);
      sqw_y(unit_cell_atom, time_point_counter_, i) = norm*complex<double>(sq_y(q.x, q.y, q.z)[0],sq_y(q.x, q.y, q.z)[1]);
      sqw_z(unit_cell_atom, time_point_counter_, i) = norm*complex<double>(sq_z(q.x, q.y, q.z)[0],sq_z(q.x, q.y, q.z)[1]);
    }
  }

  time_point_counter_++;
}

double StructureFactorMonitor::fft_windowing(const int n, const int n_total) {
  return 0.54 - 0.46*cos((kTwoPi*n)/double(n_total-1));
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

  for (int unit_cell_atom = 0; unit_cell_atom < lattice.num_unit_cell_positions(); ++unit_cell_atom) {
    for (int i = 0; i < time_points; ++i) {
      for (int j = 0; j < space_points; ++j) {
        fft_sqw_x(i,j)[0] = sqw_x(unit_cell_atom, i, j).real()*fft_windowing(i, time_points);
        fft_sqw_x(i,j)[1] = sqw_x(unit_cell_atom, i, j).imag()*fft_windowing(i, time_points);

        fft_sqw_y(i,j)[0] = sqw_y(unit_cell_atom, i, j).real()*fft_windowing(i, time_points);
        fft_sqw_y(i,j)[1] = sqw_y(unit_cell_atom, i, j).imag()*fft_windowing(i, time_points);

        fft_sqw_z(i,j)[0] = sqw_z(unit_cell_atom, i, j).real()*fft_windowing(i, time_points);
        fft_sqw_z(i,j)[1] = sqw_z(unit_cell_atom, i, j).imag()*fft_windowing(i, time_points);
      }
    }

    fftw_execute(fft_plan_time_x);
    fftw_execute(fft_plan_time_y);
    fftw_execute(fft_plan_time_z);

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

  std::string name = seedname + "_dsf.dat";
  std::ofstream dsffile(name.c_str());

  for (int i = 0; i < (time_points/2) + 1; ++i) {
    double total_length = 0.0;
    for (int j = 0; j < space_points; ++j) {
      dsffile << j << "\t" << total_length << "\t" << i*freq_delta << "\t";
      dsffile << total_mag_sqw_x(i,j) << "\t" << total_sqw_x(i,j)[0] << "\t" << total_sqw_x(i,j)[1] << "\t";
      dsffile << total_mag_sqw_y(i,j) << "\t" << total_sqw_y(i,j)[0] << "\t" << total_sqw_y(i,j)[1] << "\t";
      dsffile << total_mag_sqw_z(i,j) << "\t" << total_sqw_z(i,j)[0] << "\t" << total_sqw_z(i,j)[1] << "\n";
      total_length += bz_lengths[j];
    }
    dsffile << std::endl;
  }

  dsffile.close();

  // spin pumping
  fftw_plan chi_xy_plan = fftw_plan_dft_1d(time_points, chi_xy.data(), chi_xy.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(chi_xy_plan);

  // spin pumping
  fftw_plan chi_yx_plan = fftw_plan_dft_1d(time_points, chi_yx.data(), chi_yx.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(chi_yx_plan);



  name = seedname + "_chi.tsv";
  std::ofstream chifile(name.c_str());

  for (int i = 0; i < (time_points/2) + 1; ++i) {
    chifile << i*freq_delta << "\t" << chi_xy[i][0] << "\t" << chi_xy[i][1] << "\t" << chi_yx[i][0] << "\t" << chi_yx[i][1] << "\n";
  }
  chifile.close();
}

StructureFactorMonitor::~StructureFactorMonitor() {
  fft_time();
}
