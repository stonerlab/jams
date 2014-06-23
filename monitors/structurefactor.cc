// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

#include <fftw3.h>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/structurefactor.h"

// We can't guarenttee that FFT methods are being used by the integrator, so we implement all of the FFT
// with the monitor. This may mean performing the FFT twice, but presumably the structure factor is being
// calculated much less frequently than every integration step.

StructureFactorMonitor::StructureFactorMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Structure Factor monitor...\n");

  is_equilibration_monitor_ = false;

  // plan FFTW routines
  sq_xy.resize(lattice.kspace_size(0), lattice.kspace_size(1), lattice.kspace_size(2));
  fft_plan_sq_xy = fftw_plan_dft_3d(lattice.kspace_size(0), lattice.kspace_size(1), lattice.kspace_size(2), sq_xy.data(),  sq_xy.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  sq_z.resize(lattice.kspace_size(0), lattice.kspace_size(1), lattice.kspace_size(2));
  fft_plan_sq_z = fftw_plan_dft_3d(lattice.kspace_size(0), lattice.kspace_size(1), lattice.kspace_size(2), sq_z.data(),  sq_z.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  // create transform arrays for example to apply a Holstein Primakoff transform
  s_transform.resize(num_spins, 3);

  libconfig::Setting& material_settings = ::config.lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      s_transform(i,n) = material_settings[::lattice.get_material_number(i)]["transform"][n];
    }
  }

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
      if (int(bz_nodes[n][i]) > (lattice.kspace_size(i)/2 + 1)) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n][0]), int(bz_nodes[n][1]), int(bz_nodes[n][2]));
      }
      if (int(bz_nodes[n+1][i]) > (lattice.kspace_size(i)/2 + 1)) {
        jams_error("bz node point [ %4d %4d %4d ] is larger than the kspace", int(bz_nodes[n+1][0]), int(bz_nodes[n+1][1]), int(bz_nodes[n+1][2]));
      }
    }

    // vector between the nodes
    for (int i = 0; i < 3; ++i) {
      bz_line[i] = int(bz_nodes[n+1][i]) - int(bz_nodes[n][i]);
    }
    if (verbose_output_is_set) {::output.write("  bz line: [ %4d %4d %4d ]\n", bz_line.x, bz_line.y, bz_line.z); }

    // normalised vector
    for (int i = 0; i < 3; ++i) {
      bz_line[i] != 0 ? bz_line_element[i] = bz_line[i]/abs(bz_line[i]) : bz_line_element[i] = 0;
    }

    // the number of points is the max dimension in line
    const int bz_line_points = abs(*std::max_element(bz_line.begin(), bz_line.end(), [] (int a, int b) { return (abs(a) < abs(b)); }));
    if (verbose_output_is_set) { ::output.write("  bz line points: %d\n", bz_line_points); }

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
}

void StructureFactorMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  // zero the sq arrays
  for (int i = 0; i < sq_xy.elements(); ++i) {
    sq_xy[i][0] = 0.0; sq_xy[i][1] = 0.0;
    sq_z[i][0] = 0.0;  sq_z[i][1] = 0.0;
  }

  double mz = 0.0;

  for (int i = 0; i < num_spins; ++i) {
    mz += s(i,2)*s_transform(i,2);
  }
  mz /= double(num_spins);

  // remap spin data into kspace array
  for (int i = 0; i < num_spins; ++i) {
    jblib::Vec3<int> r(::lattice.kspace_inv_map_(i,0), ::lattice.kspace_inv_map_(i,1), ::lattice.kspace_inv_map_(i,2));
    sq_xy(r.x, r.y, r.z)[0] = s(i,0)*s_transform(i,0);     sq_xy(r.x, r.y, r.z)[1] = s(i,1)*s_transform(i,1);
    sq_z(r.x, r.y, r.z)[0]  = s(i,2)*s_transform(i,2)-mz;  sq_z(r.x, r.y, r.z)[1]  = 0.0;
  }

  // perform in place FFT
  fftw_execute(fft_plan_sq_xy); fftw_execute(fft_plan_sq_z);

  // add the Sq to the timeseries
  for (int i = 0, iend = bz_points.size(); i < iend; ++i) {
    jblib::Vec3<int> q = bz_points[i];
    sqw_xy.push_back(std::complex<double>(sq_xy(q.x, q.y, q.z)[0],sq_xy(q.x, q.y, q.z)[1]));
    sqw_z.push_back(std::complex<double>(sq_z(q.x, q.y, q.z)[0],sq_z(q.x, q.y, q.z)[1]));
  }
}

double StructureFactorMonitor::fft_windowing(const int n, const int n_total) {
  return 0.54 - 0.46*cos((2.0*M_PI*n)/double(n_total-1));
}

void StructureFactorMonitor::fft_time() {

  const int space_points = bz_points.size();
  const int time_points = sqw_xy.size()/space_points;

  jblib::Array<fftw_complex,2> fft_sqw_xy(time_points, space_points);
  jblib::Array<fftw_complex,2> fft_sqw_z(time_points, space_points);

  int rank       = 1;
  int sizeN[]   = {time_points};
  int howmany    = space_points;
  int inembed[] = {time_points}; int onembed[] = {time_points};
  int istride    = space_points; int ostride    = space_points;
  int idist      = 1;            int odist      = 1;

  fftw_plan fft_plan_time_xy = fftw_plan_many_dft(rank,sizeN,howmany,fft_sqw_xy.data(),inembed,istride,idist,fft_sqw_xy.data(),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan fft_plan_time_z = fftw_plan_many_dft(rank,sizeN,howmany,fft_sqw_z.data(),inembed,istride,idist,fft_sqw_z.data(),onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);

  for (int i = 0; i < time_points; ++i) {
    for (int j = 0; j < space_points; ++j) {
      fft_sqw_xy(i,j)[0] = sqw_xy[i*space_points + j].real()*fft_windowing(i, time_points);
      fft_sqw_xy(i,j)[1] = sqw_xy[i*space_points + j].imag()*fft_windowing(i, time_points);

      fft_sqw_z(i,j)[0] = sqw_z[i*space_points + j].real()*fft_windowing(i, time_points);
      fft_sqw_z(i,j)[1] = sqw_z[i*space_points + j].imag()*fft_windowing(i, time_points);
    }
  }

  fftw_execute(fft_plan_time_xy);
  fftw_execute(fft_plan_time_z);

  std::string name = seedname + "_dsf.dat";
  std::ofstream dsffile(name.c_str());

  for (int i = 0; i < (time_points/2) + 1; ++i) {
    for (int j = 0; j < space_points; ++j) {
      dsffile << j << "\t" << i << "\t" << fft_sqw_xy(i,j)[0]*fft_sqw_xy(i,j)[0] + fft_sqw_xy(i,j)[1]*fft_sqw_xy(i,j)[1];
      dsffile << "\t" << fft_sqw_z(i,j)[0]*fft_sqw_z(i,j)[0] + fft_sqw_z(i,j)[1]*fft_sqw_z(i,j)[1] << std::endl;
    }
    dsffile << std::endl;
  }

  dsffile.close();
}

StructureFactorMonitor::~StructureFactorMonitor() {
  fft_time();
}
