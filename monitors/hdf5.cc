// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "H5Cpp.h"

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/hdf5.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings) {
    using namespace globals;

    ::output.write("\nInitialising HDF5 monitor...\n");

    is_equilibration_monitor_ = false;
    output_step_freq_ = settings["output_steps"];
}

void Hdf5Monitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;
  using namespace H5;

  if (iteration%output_step_freq_ == 0) {
    int outcount = iteration/output_step_freq_;  // int divisible by modulo above

    const std::string filename(seedname+"_"+zero_pad_number(outcount)+".h5");

    H5File outfile(filename.c_str(), H5F_ACC_TRUNC);

    hsize_t dims[2] = {static_cast<hsize_t>(num_spins), 3};
    DataSpace dataspace(2, dims);

    DataSet dataset = outfile.createDataSet("spins", PredType::IEEE_F64LE, dataspace);

    dataset.write(s.data(), PredType::IEEE_F64LE);


  }
}

Hdf5Monitor::~Hdf5Monitor() {
}
