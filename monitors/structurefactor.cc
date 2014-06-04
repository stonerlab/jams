// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

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

  std::string name = seedname + "_ssf.dat";
  outfile.open(name.c_str());

}

void StructureFactorMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  // zero the sq_xy array

  for (int i = 0; i < sq_xy.elements(); ++i) {
    sq_xy[i][0] = 0.0;
    sq_xy[i][1] = 0.0;
  }

  // remap spin data into kspace array
  for (int i = 0; i < num_spins; ++i) {
    int x = lattice.kspace_inv_map_(i,0);
    int y = lattice.kspace_inv_map_(i,1);
    int z = lattice.kspace_inv_map_(i,2);

    sq_xy(x,y,z)[0] = s(i,0);
    sq_xy(x,y,z)[1] = s(i,1);
  }

  // perform in place FFT
  fftw_execute(fft_plan_sq_xy);

  // add the Sq to the timeseries

for (int i = 0; i < lattice.kspace_size(2)/2 +1; ++i) {
  sqw.push_back(std::complex<double>(sq_xy(0, 0, i)[0],sq_xy(0, 0, i)[1]));
}

  // outfile << 0 << "\t" << sq_xy(0, 0, 0)[0] << "\t" << sq_xy(0, 0, 0)[1] << "\n";
  // for (int i = 1; i < lattice.kspace_size(2)/2; ++i) {
  //   sqw.push_back(std::complex<double>(sq_xy(0, 0, i)[0],sq_xy(0, 0, i)[1]));

  //   fftw_complex sq_plus =  {sq_xy(0, 0, i)[0], sq_xy(0, 0, i)[1]};
  //   fftw_complex sq_minus = {sq_xy(0, 0, lattice.kspace_size(0)-i)[0], sq_xy(0, 0, lattice.kspace_size(0)-i)[1]};

  //   fftw_complex convolution = {sq_minus[0]*sq_plus[0] - sq_minus[1]*sq_plus[1],
  //                               sq_minus[0]*sq_plus[1] + sq_minus[1]*sq_plus[0]};

  //   outfile << i << "\t" << convolution[0] << "\t" << convolution[1] << "\n";
  // }

  // outfile << "\n" << std::endl;
}

double StructureFactorMonitor::fft_windowing(const int n, const int n_total) {
  return 0.54 - 0.46*cos((2.0*M_PI*n)/double(n_total-1));
}

void StructureFactorMonitor::fft_time() {

  const int space_points = (lattice.kspace_size(2)/2) + 1;
  const int time_points = sqw.size()/space_points;

  jblib::Array<fftw_complex,2> fft_sqw(time_points, space_points);

  int rank       = 1;
  int sizeN[]   = {time_points};
  int howmany    = space_points;
  int inembed[] = {time_points}; int onembed[] = {time_points};
  int istride    = space_points; int ostride    = space_points;
  int idist      = 1;            int odist      = 1;
  fftw_complex* startPtr = fft_sqw.data();

  fftw_plan fft_plan_time = fftw_plan_many_dft(rank,sizeN,howmany,startPtr,inembed,istride,idist,startPtr,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);

  for (int i = 0; i < time_points; ++i) {
    for (int j = 0; j < space_points; ++j) {
      fft_sqw(i,j)[0] = sqw[i*space_points + j].real()*fft_windowing(i, time_points);
      fft_sqw(i,j)[1] = sqw[i*space_points + j].imag()*fft_windowing(i, time_points);
    }
  }

  // fftw_plan fft_plan_time = fftw_plan_dft(2, n, fft_sqw.data(), fft_sqw.data(), FFTW_FORWARD, FFTW_ESTIMATE);

  for (int i = 0; i < sqw.size(); ++i){
    fft_sqw[i][0] = sqw[i].real();
    fft_sqw[i][1] = sqw[i].imag();
  }

  fftw_execute(fft_plan_time);

  std::string name = seedname + "_dsf.dat";
  std::ofstream dsffile(name.c_str());

  for (int i = 0; i < (time_points/2) + 1; ++i) {
    for (int j = 0; j < space_points; ++j) {
      dsffile << j << "\t" << i << "\t" << fft_sqw(i,j)[0]*fft_sqw(i,j)[0] + fft_sqw(i,j)[1]*fft_sqw(i,j)[1]<< std::endl;
    }
    dsffile << std::endl;
  }

  dsffile.close();



  // const int space_points = (lattice.kspace_size(2)/2) + 1;
  // const int time_points = sqw.size()/space_points;

  // jblib::Array<fftw_complex,2> fft_sqw(time_points, space_points);

  // int rank       = 1;
  // int sizeN[]   = {time_points};
  // int howmany    = space_points;
  // int inembed[] = {time_points}; int onembed[] = {time_points};
  // int istride    = space_points; int ostride    = space_points;
  // int idist      = 1;            int odist      = 1;
  // fftw_complex* startPtr = fft_sqw.data();

  // fftw_plan fft_plan_time = fftw_plan_many_dft(rank,sizeN,howmany,startPtr,inembed,istride,idist,startPtr,onembed,ostride,odist,FFTW_FORWARD,FFTW_ESTIMATE);

  // for (int i = 0; i < time_points; ++i) {
  //   for (int j = 0; j < space_points; ++j) {
  //     fft_sqw(i,j)[0] = sqw[i*space_points + j].real()*fft_windowing(i, time_points);
  //     fft_sqw(i,j)[1] = sqw[i*space_points + j].imag()*fft_windowing(i, time_points);
  //   }
  // }

  // fftw_execute(fft_plan_time);

  // std::string name = seedname + "_dsf.dat";
  // std::ofstream dsffile(name.c_str());


  // for (int i = 0; i < (time_points/2) + 1; ++i) {
  //   for (int j = 0; j < space_points; ++j) {
  //     dsffile << j << "\t" << i << "\t" << fft_sqw(i,j)[0]*fft_sqw(i,j)[0] + fft_sqw(i,j)[1]*fft_sqw(i,j)[1]<< std::endl;
  //   }
  //   dsffile << std::endl;
  // }

  // dsffile.close();

}

StructureFactorMonitor::~StructureFactorMonitor() {
  fft_time();
  outfile.close();
}
