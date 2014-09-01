// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <algorithm>

#include "H5Cpp.h"

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/hdf5.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

namespace {
    const hsize_t h5_compression_chunk_size = 10000;
    const hsize_t h5_compression_factor = 6;
}

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings),
  float_pred_type(H5::PredType::IEEE_F64LE),
  is_compression_enabled(false) {
    using namespace globals;
    using namespace H5;

    ::output.write("\nInitialising HDF5 monitor...\n");

    is_equilibration_monitor_ = false;
    output_step_freq_ = settings["output_steps"];

    // floating point output precision
    if (settings.exists("float_type")) {
        if(capitalize(settings["float_type"]) == "FLOAT") {
            float_pred_type = PredType::IEEE_F32LE;
            ::output.write("  float data stored as float (IEEE_F32LE)\n");
        } else if(capitalize(settings["float_type"]) == "DOUBLE") {
            float_pred_type = PredType::IEEE_F64LE;
            ::output.write("  float data stored as double (IEEE_F64LE)\n");
        } else {
            jams_error("Unknown float_type selected for HDF5 monitor.\nOptions: float or double");
        }
    } else {
        ::output.write("  float data stored as double (IEEE_F64LE)\n");
    }

    // compression options
    settings.lookupValue("compressed", is_compression_enabled);
    ::output.write("  compressed: %s\n", is_compression_enabled ? "enabled": "disabled");

    // output lattice
    output_lattice();

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

    DSetCreatPropList plist;

    if (is_compression_enabled) {
        hsize_t chunk_dims[2] = {std::min(h5_compression_chunk_size, static_cast<hsize_t>(num_spins)), 3};
        plist.setChunk(2, chunk_dims);
        plist.setDeflate(h5_compression_factor);
    }

    DataSet dataset = outfile.createDataSet("spins", float_pred_type, dataspace, plist);

    dataset.write(s.data(), PredType::NATIVE_DOUBLE);
  }
}

void Hdf5Monitor::output_lattice() {
    using namespace H5;
    using namespace globals;

    const std::string filename(seedname+"_lattice.h5");
    H5File outfile(filename.c_str(), H5F_ACC_TRUNC);

    hsize_t type_dims[1] = {static_cast<hsize_t>(num_spins)};
    DataSpace types_dataspace(1, type_dims);
    DataSet types_dataset = outfile.createDataSet("types", PredType::STD_I32LE, types_dataspace);
    types_dataset.write(&lattice.lattice_material_num_[0], PredType::NATIVE_INT32);

    jblib::Array<double, 2> lattice_pos(num_spins, 3);
    for (int i = 0; i < num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
            lattice_pos(i,j) = lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
        }
    }
    hsize_t pos_dims[2] = {static_cast<hsize_t>(num_spins), 3};
    DataSpace pos_dataspace(2, pos_dims);
    DataSet pos_dataset = outfile.createDataSet("positions", float_pred_type, pos_dataspace);
    pos_dataset.write(lattice_pos.data(), PredType::NATIVE_DOUBLE);
}

Hdf5Monitor::~Hdf5Monitor() {
}
