#include <complex>

#include <fftw3.h>

#include "jams/core/globals.h"
#include "jams/core/consts.h"

#include "jams/core/field.h"


using std::complex;
using jblib::Array;

void fft_scalar_field(
	const Array<double, 1>& field, // input field array layed out the same as the spin array
	Array<complex<double>, 4>& output) 	  // FFT'd 3D array where 4th dim is the unitcell position
{
	using namespace globals;

	const double norm = 1.0/sqrt(product(lattice.kspace_size()));

	complex<double> two_pi_i_dr;
	complex<double> exp_phase_0;
	Array<complex<double>, 1> exp_phase_x(lattice.kspace_size().x);
	Array<complex<double>, 1> exp_phase_y(lattice.kspace_size().y);
	Array<complex<double>, 1> exp_phase_z(lattice.kspace_size().z);

	Array<complex<double>, 3> remapped_field(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);

	output.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, lattice.num_unit_cell_positions());
	output.zero();

	fftw_plan plan = fftw_plan_dft_3d(
		lattice.kspace_size().x, 
		lattice.kspace_size().y, 
		lattice.kspace_size().z, 
		reinterpret_cast<fftw_complex*>(remapped_field.data()),  
		reinterpret_cast<fftw_complex*>(remapped_field.data()), 
		FFTW_FORWARD, 
		FFTW_ESTIMATE);
	
	for (int n = 0; n < lattice.num_unit_cell_positions(); ++n) {
		remapped_field.zero();

		for (int i = 0; i < num_spins; ++i) {
		  if ((i+n)%lattice.num_unit_cell_positions() == 0) {
		    jblib::Vec3<int> r = lattice.super_cell_pos(i);
		    remapped_field(r.x, r.y, r.z) = {field(i), 0.0};
		  }
		}

		fftw_execute(plan);

		// calculate phase factors

		// super speed hack from CASTEP ewald.f90 for generating all of the phase factors on
		// the fly without calling lots of exp()
		two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(n).x;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_x(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().x - 1)));
		for (int i = 1; i < lattice.kspace_size().x; ++i) {
		  exp_phase_x(i) = exp_phase_x(i-1)*exp_phase_0;
		}

		two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(n).y;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_y(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().y - 1)));
		for (int i = 1; i < lattice.kspace_size().y; ++i) {
		  exp_phase_y(i) = exp_phase_y(i-1)*exp_phase_0;
		}

		two_pi_i_dr = kImagTwoPi* lattice.unit_cell_position_cart(n).z;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_z(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().z - 1)));
		for (int i = 1; i < lattice.kspace_size().z; ++i) {
		  exp_phase_z(i) = exp_phase_z(i-1)*exp_phase_0;
		}

		// normalize the transform
		for (int i = 0; i < lattice.kspace_size().x; ++i) {
		  for (int j = 0; j < lattice.kspace_size().y; ++j) {
		    for (int k = 0; k < lattice.kspace_size().z; ++k) {
		      output(i, j, k, n) = remapped_field(i, j, k) * norm * exp_phase_x(i) * exp_phase_y(j) * exp_phase_z(k);
		    }
		  }
		}
	}

	fftw_destroy_plan(plan);
}

//---------------------------------------------------------------------

void fft_vector_field(
	const jblib::Array<double, 2>& field, 
	jblib::Array<complex<double>, 4>& out_x,
	jblib::Array<complex<double>, 4>& out_y,
	jblib::Array<complex<double>, 4>& out_z)
{
	using namespace globals;

	const double norm = 1.0/sqrt(product(lattice.kspace_size()));

	complex<double> two_pi_i_dr;
	complex<double> exp_phase_0;
	Array<complex<double>, 1> exp_phase_x(lattice.kspace_size().x);
	Array<complex<double>, 1> exp_phase_y(lattice.kspace_size().y);
	Array<complex<double>, 1> exp_phase_z(lattice.kspace_size().z);

	Array<complex<double>, 3> remapped_x(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);
	Array<complex<double>, 3> remapped_y(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);
	Array<complex<double>, 3> remapped_z(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z);

	out_x.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, lattice.num_unit_cell_positions());
	out_y.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, lattice.num_unit_cell_positions());
	out_z.resize(lattice.kspace_size().x, lattice.kspace_size().y, lattice.kspace_size().z, lattice.num_unit_cell_positions());

	out_x.zero();
	out_y.zero();
	out_z.zero();

	fftw_plan plan_x = fftw_plan_dft_3d(
		lattice.kspace_size().x, 
		lattice.kspace_size().y, 
		lattice.kspace_size().z, 
		reinterpret_cast<fftw_complex*>(remapped_x.data()),  
		reinterpret_cast<fftw_complex*>(remapped_x.data()), 
		FFTW_FORWARD, 
		FFTW_ESTIMATE);

	fftw_plan plan_y = fftw_plan_dft_3d(
		lattice.kspace_size().x, 
		lattice.kspace_size().y, 
		lattice.kspace_size().z, 
		reinterpret_cast<fftw_complex*>(remapped_y.data()),  
		reinterpret_cast<fftw_complex*>(remapped_y.data()), 
		FFTW_FORWARD, 
		FFTW_ESTIMATE);

	fftw_plan plan_z = fftw_plan_dft_3d(
		lattice.kspace_size().x, 
		lattice.kspace_size().y, 
		lattice.kspace_size().z, 
		reinterpret_cast<fftw_complex*>(remapped_z.data()),  
		reinterpret_cast<fftw_complex*>(remapped_z.data()), 
		FFTW_FORWARD, 
		FFTW_ESTIMATE);
	
	for (int n = 0; n < lattice.num_unit_cell_positions(); ++n) {
		remapped_x.zero();
		remapped_y.zero();
		remapped_z.zero();

		for (int i = 0; i < num_spins; ++i) {
		  if ((i+n)%lattice.num_unit_cell_positions() == 0) {
		    jblib::Vec3<int> r = lattice.super_cell_pos(i);
		    remapped_x(r.x, r.y, r.z) = {field(i, 0), 0.0};
		    remapped_y(r.x, r.y, r.z) = {field(i, 1), 0.0};
		    remapped_z(r.x, r.y, r.z) = {field(i, 2), 0.0};
		  }
		}

		fftw_execute(plan_x);
		fftw_execute(plan_y);
		fftw_execute(plan_z);

		// calculate phase factors

		// super speed hack from CASTEP ewald.f90 for generating all of the phase factors on
		// the fly without calling lots of exp()
		two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(n).x;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_x(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().x - 1)));
		for (int i = 1; i < lattice.kspace_size().x; ++i) {
		  exp_phase_x(i) = exp_phase_x(i-1)*exp_phase_0;
		}

		two_pi_i_dr = kImagTwoPi*lattice.unit_cell_position_cart(n).y;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_y(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().y - 1)));
		for (int i = 1; i < lattice.kspace_size().y; ++i) {
		  exp_phase_y(i) = exp_phase_y(i-1)*exp_phase_0;
		}

		two_pi_i_dr = kImagTwoPi* lattice.unit_cell_position_cart(n).z;
		exp_phase_0 = exp(two_pi_i_dr);
		exp_phase_z(0) = exp(-two_pi_i_dr*double((lattice.kspace_size().z - 1)));
		for (int i = 1; i < lattice.kspace_size().z; ++i) {
		  exp_phase_z(i) = exp_phase_z(i-1)*exp_phase_0;
		}

		// normalize the transform
		for (int i = 0; i < lattice.kspace_size().x; ++i) {
		  for (int j = 0; j < lattice.kspace_size().y; ++j) {
		    for (int k = 0; k < lattice.kspace_size().z; ++k) {
		      out_x(i, j, k, n) = remapped_x(i, j, k) * norm * exp_phase_x(i) * exp_phase_y(j) * exp_phase_z(k);
		      out_y(i, j, k, n) = remapped_y(i, j, k) * norm * exp_phase_x(i) * exp_phase_y(j) * exp_phase_z(k);
		      out_z(i, j, k, n) = remapped_z(i, j, k) * norm * exp_phase_x(i) * exp_phase_y(j) * exp_phase_z(k);
		    }
		  }
		}
	}

	fftw_destroy_plan(plan_x);
	fftw_destroy_plan(plan_y);
	fftw_destroy_plan(plan_z);
}
