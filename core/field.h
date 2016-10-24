// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_FIELD_H
#define JAMS_CORE_FIELD_H

#include <complex>

#include "jblib/containers/array.h"

void fft_scalar_field(const jblib::Array<double, 1>& field, jblib::Array<std::complex<double>, 4>& output);

void fft_vector_field(
	const jblib::Array<double, 2>& field, 
	jblib::Array<std::complex<double>, 4>& out_x,
	jblib::Array<std::complex<double>, 4>& out_y,
	jblib::Array<std::complex<double>, 4>& out_z);

#endif  // JAMS_CORE_FIELD_H