// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdio>
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
  is_compression_enabled(false),
  slice() {
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

    // create xdmf_file
    const std::string xdmf_filename(seedname+".xdmf");
    xdmf_file = fopen(xdmf_filename.c_str(), "w");

                 fputs("<?xml version=\"1.0\"?>\n", xdmf_file);
                 fputs("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\"[]>\n", xdmf_file);
                 fputs("<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n", xdmf_file);
                 fputs("  <Domain Name=\"JAMS\">\n", xdmf_file);
    fprintf(xdmf_file, "    <Information Name=\"Commit\" Value=\"%s\" />\n", QUOTEME(GITCOMMIT));
    fprintf(xdmf_file, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", seedname.c_str());
                 fputs("    <Grid Name=\"Time\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file);
                 fputs("    </Grid>\n", xdmf_file);
                 fputs("  </Domain>\n", xdmf_file);
                 fputs("</Xdmf>", xdmf_file);
                 fflush(xdmf_file);
    // output lattice
    output_lattice();
}

Hdf5Monitor::~Hdf5Monitor() {
    fclose(xdmf_file);
}

void Hdf5Monitor::update(const Solver * const solver) {
  using namespace globals;
  using namespace H5;

  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

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

    double out_iteration = solver->iteration();
    double out_time = solver->time();
    double out_temperature = solver->physics()->temperature();
    jblib::Vec3<double> out_field = solver->physics()->applied_field();

    DataSet spin_dataset = outfile.createDataSet("spins", float_pred_type, dataspace, plist);

    DataSpace attribute_dataspace(H5S_SCALAR);
    Attribute attribute = spin_dataset.createAttribute("iteration", PredType::STD_I32LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_INT32, &out_iteration);
    attribute = spin_dataset.createAttribute("time", PredType::IEEE_F64LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_DOUBLE, &out_time);
    attribute = spin_dataset.createAttribute("temperature", PredType::IEEE_F64LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_DOUBLE, &out_temperature);
    attribute = spin_dataset.createAttribute("hx", PredType::IEEE_F64LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_DOUBLE, &out_field.x);
    attribute = spin_dataset.createAttribute("hy", PredType::IEEE_F64LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_DOUBLE, &out_field.y);
    attribute = spin_dataset.createAttribute("hz", PredType::IEEE_F64LE, attribute_dataspace);
    attribute.write(PredType::NATIVE_DOUBLE, &out_field.z);

    if (slice.num_points() != 0) {
        jblib::Array<double, 2> spin_slice(slice.num_points(), 3);
        for (int i = 0; i < slice.num_points(); ++i) {
            for (int j = 0; j < 3; ++j) {
                spin_slice(i,j) = slice.spin(i, j);
            }
        }
        spin_dataset.write(spin_slice.data(), PredType::NATIVE_DOUBLE);
    } else {
        spin_dataset.write(s.data(), PredType::NATIVE_DOUBLE);
    }



                 // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
                 fseek(xdmf_file, -31, SEEK_CUR);

                 fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file);
    fprintf(xdmf_file, "        <Time Value=\"%f\" />\n", out_time/1e-12);
    fprintf(xdmf_file, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%llu\" />\n", dims[0]);
                 fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file);
    if (float_pred_type == PredType::IEEE_F32LE) {
    fprintf(xdmf_file, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", dims[0]);
    } else {
    fprintf(xdmf_file, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", dims[0]);
    }
    fprintf(xdmf_file, "           %s_lattice.h5:/positions\n", seedname.c_str());
                 fputs("         </DataItem>\n", xdmf_file);
                 fputs("       </Geometry>\n", xdmf_file);
                 fputs("       <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file);
    fprintf(xdmf_file, "         <DataItem Dimensions=\"%llu\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n", dims[0]);
    fprintf(xdmf_file, "           %s_lattice.h5:/types\n", seedname.c_str());
                 fputs("         </DataItem>\n", xdmf_file);
                 fputs("       </Attribute>\n", xdmf_file);
                 fputs("       <Attribute Name=\"Spin\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file);
    if (float_pred_type == PredType::IEEE_F32LE) {
    fprintf(xdmf_file, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", dims[0]);
    } else {
    fprintf(xdmf_file, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", dims[0]);
    }
    fprintf(xdmf_file, "           %s:/spins\n", filename.c_str());
                 fputs("         </DataItem>\n", xdmf_file);
                 fputs("       </Attribute>\n", xdmf_file);
                 fputs("      </Grid>\n", xdmf_file);

                 // reprint the closing tags of the XML
                 fputs("    </Grid>\n", xdmf_file);
                 fputs("  </Domain>\n", xdmf_file);
                 fputs("</Xdmf>", xdmf_file);
    fflush(xdmf_file);
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
            types(i) = lattice.material(i);
        }

        positions.resize(num_spins, 3);

        for (int i = 0; i < pos_dims[0]; ++i) {
            for (int j = 0; j < 3; ++j) {
               positions(i, j) = lattice.parameter()*lattice.position(i)[j];
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
