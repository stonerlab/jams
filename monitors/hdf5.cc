// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <algorithm>

#include "H5Cpp.h"

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"
#include "core/slice.h"

#include "monitors/hdf5.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

namespace {
    const hsize_t h5_compression_chunk_size = 256;
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
        if (capitalize(settings["float_type"]) == "FLOAT") {
            float_pred_type = PredType::IEEE_F32LE;
            ::output.write("  float data stored as float (IEEE_F32LE)\n");
        } else if (capitalize(settings["float_type"]) == "DOUBLE") {
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

    if (settings.exists("slice")) {
        slice = Slice(settings["slice"]);
    }

    // output lattice
    output_lattice();
}

void Hdf5Monitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;
  using namespace H5;

  if (iteration%output_step_freq_ == 0) {
    int outcount = iteration/output_step_freq_;  // int divisible by modulo above

    hsize_t dims[2], chunk_dims[2];

    const std::string filename(seedname+"_"+zero_pad_number(outcount)+".h5");

    H5File outfile(filename.c_str(), H5F_ACC_TRUNC);

    if (slice.num_points() != 0) {
        dims[0] = static_cast<hsize_t>(slice.num_points());
        dims[1] = 3;
        chunk_dims[0] = std::min(h5_compression_chunk_size, static_cast<hsize_t>(slice.num_points()));
        chunk_dims[1] = 3;
    } else {
        dims[0] = static_cast<hsize_t>(num_spins);
        dims[1] = 3;
        chunk_dims[0] = std::min(h5_compression_chunk_size, static_cast<hsize_t>(num_spins));
        chunk_dims[1] = 3;
    }

    DataSpace dataspace(2, dims);

    DSetCreatPropList plist;

    if (is_compression_enabled) {
        plist.setChunk(2, chunk_dims);
        plist.setDeflate(h5_compression_factor);
    }

    DataSet dataset = outfile.createDataSet("spins", float_pred_type, dataspace, plist);

    if (slice.num_points() != 0) {
        jblib::Array<double, 2> spin_slice(slice.num_points(), 3);
        for (int i = 0; i < slice.num_points(); ++i) {
            for (int j = 0; j < 3; ++j) {
                spin_slice(i,j) = slice.spin(i, j);
            }
        }
        dataset.write(spin_slice.data(), PredType::NATIVE_DOUBLE);
    } else {
        dataset.write(s.data(), PredType::NATIVE_DOUBLE);
    }
  }
}

void Hdf5Monitor::output_lattice() {
    using namespace H5;
    using namespace globals;

    hsize_t type_dims[1], pos_dims[2];

    jblib::Array<int, 1>    types;
    jblib::Array<double, 2> positions;

    const std::string filename(seedname+"_lattice.h5");
    H5File outfile(filename.c_str(), H5F_ACC_TRUNC);

    if (slice.num_points() != 0) {
        type_dims[0] = static_cast<hsize_t>(slice.num_points());
        types.resize(slice.num_points());

        for (int i = 0; i < type_dims[0]; ++i) {
            types(i) = slice.type(i);
        }

        pos_dims[0]  = static_cast<hsize_t>(slice.num_points());
        pos_dims[1]  = 3;

        positions.resize(slice.num_points(), 3);

        for (int i = 0; i < pos_dims[0]; ++i) {
            for (int j = 0; j < 3; ++j) {
               positions(i, j) = slice.position(i, j);
            }
        }
    } else {
        type_dims[0] = static_cast<hsize_t>(num_spins);
        pos_dims[0]  = static_cast<hsize_t>(num_spins);
        pos_dims[1]  = 3;

        types.resize(num_spins);

        for (int i = 0; i < type_dims[0]; ++i) {
            types(i) = lattice.lattice_material_num_[i];
        }

        pos_dims[0]  = static_cast<hsize_t>(slice.num_points());
        pos_dims[1]  = 3;

        positions.resize(num_spins, 3);

        for (int i = 0; i < pos_dims[0]; ++i) {
            for (int j = 0; j < 3; ++j) {
               positions(i, j) = lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
            }
        }
    }

    DataSpace types_dataspace(1, type_dims);
    DataSet types_dataset = outfile.createDataSet("types", PredType::STD_I32LE, types_dataspace);
    types_dataset.write(types.data(), PredType::NATIVE_INT32);

    DataSpace pos_dataspace(2, pos_dims);
    DataSet pos_dataset = outfile.createDataSet("positions", float_pred_type, pos_dataspace);
    pos_dataset.write(positions.data(), PredType::NATIVE_DOUBLE);
}

Hdf5Monitor::~Hdf5Monitor() {
}
