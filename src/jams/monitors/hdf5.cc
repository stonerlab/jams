// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdio>
#include <climits>
#include <string>
#include <algorithm>
#include <vector>

#include "version.h"
#include "H5Cpp.h"

#include "jams/helpers/error.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/slice.h"
#include "jams/interface/h5.h"

#include "hdf5.h"

using namespace std;

namespace {
    const unsigned h5_compression_chunk_size = 4095;
    const unsigned h5_compression_factor = 6;
}

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings),
  float_pred_type_(H5::PredType::IEEE_F64LE),
  slice_() {
    using namespace globals;
    using namespace H5;

    output_step_freq_ = settings["output_steps"];

    // it outsteps is 0 then use max int instead - output will only be generated in the
    // constructor and destructor
    if (output_step_freq_ == 0){
      output_step_freq_ = INT_MAX;
    }

    // floating point output precision
    if (settings.exists("float_type")) {
        if (capitalize(settings["float_type"]) == "FLOAT") {
            float_pred_type_ = PredType::IEEE_F32LE;
            cout << "  float data stored as float (IEEE_F32LE)\n";
        } else if (capitalize(settings["float_type"]) == "DOUBLE") {
            float_pred_type_ = PredType::IEEE_F64LE;
            cout << "  float data stored as double (IEEE_F64LE)\n";
        } else {
          jams_die("Unknown float_type selected for HDF5 monitor.\nOptions: float or double");
        }
    } else {
        cout << "  float data stored as double (IEEE_F64LE)\n";
    }

    // compression options
    settings.lookupValue("compressed", compression_enabled_);
    cout << "  compressed " << compression_enabled_ << "\n";

    if (settings.exists("slice")) {
        slice_ = Slice(settings["slice"]);
    }

    open_new_xdmf_file(seedname + ".xdmf");

    write_lattice_h5_file(seedname + "_lattice.h5", PredType::IEEE_F64LE);
}

Hdf5Monitor::~Hdf5Monitor() {
  // always write final in double precision
    write_spin_h5_file(seedname + "_final.h5", H5::PredType::IEEE_F64LE);
    update_xdmf_file(seedname + "_final.h5", H5::PredType::IEEE_F64LE);

    fclose(xdmf_file_);
}



void Hdf5Monitor::update(Solver * solver) {
  using namespace globals;
  using namespace H5;

  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    const std::string h5_file_name(seedname + "_" + zero_pad_number(outcount) + ".h5");

    write_spin_h5_file(h5_file_name, float_pred_type_);
    update_xdmf_file(h5_file_name, float_pred_type_);
  }
}

void Hdf5Monitor::write_spin_h5_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace HighFive;


  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  DataSetCreateProps props;

  if (compression_enabled_) {
    props.add(Chunking({std::min(h5_compression_chunk_size, num_spins), 1}));
    props.add(Shuffle());
    props.add(Deflate(h5_compression_factor));
  }

  auto dataset = file.createDataSet<double>("/spins",  DataSpace({num_spins, 3}), props);

  dataset.createAttribute("iteration", solver->iteration());
  dataset.createAttribute("time", solver->time());
  dataset.createAttribute("temperature", solver->physics()->temperature());
  dataset.createAttribute("hx", solver->physics()->applied_field()[0]);
  dataset.createAttribute("hy", solver->physics()->applied_field()[1]);
  dataset.createAttribute("hz", solver->physics()->applied_field()[2]);

  dataset.write(s);
}

//---------------------------------------------------------------------

void Hdf5Monitor::write_lattice_h5_file(const std::string &h5_file_name, const H5::PredType float_type) {
    using namespace H5;
    using namespace globals;

    hsize_t type_dims[1], pos_dims[2];

    jams::MultiArray<int, 1>    types;
    jams::MultiArray<double, 2> positions;

    H5File outfile(h5_file_name.c_str(), H5F_ACC_TRUNC);

    if (slice_.num_points() != 0) {
        type_dims[0] = static_cast<hsize_t>(slice_.num_points());
        types.resize(slice_.num_points());

        for (int i = 0; i < type_dims[0]; ++i) {
            types(i) = slice_.type(i);
        }

        pos_dims[0]  = static_cast<hsize_t>(slice_.num_points());
        pos_dims[1]  = 3;

        positions.resize(slice_.num_points(), 3);

        for (int i = 0; i < pos_dims[0]; ++i) {
            for (int j = 0; j < 3; ++j) {
               positions(i, j) = slice_.position(i, j);
            }
        }
    } else {
        type_dims[0] = static_cast<hsize_t>(num_spins);
        pos_dims[0]  = static_cast<hsize_t>(num_spins);
        pos_dims[1]  = 3;

        types.resize(num_spins);

        for (int i = 0; i < type_dims[0]; ++i) {
            types(i) = lattice->atom_material_id(i);
        }

        positions.resize(num_spins, 3);

        for (int i = 0; i < pos_dims[0]; ++i) {
            for (int j = 0; j < 3; ++j) {
               positions(i, j) = lattice->parameter()*lattice->atom_position(i)[j]/1e-9;
            }
        }
    }

    DataSpace types_dataspace(1, type_dims);
    DataSet types_dataset = outfile.createDataSet("types", PredType::STD_I32LE, types_dataspace);
    types_dataset.write(types.data(), PredType::NATIVE_INT32);

    DataSpace pos_dataspace(2, pos_dims);
    DataSet pos_dataset = outfile.createDataSet("positions", float_type, pos_dataspace);
    pos_dataset.write(positions.data(), PredType::NATIVE_DOUBLE);
}

//---------------------------------------------------------------------

void Hdf5Monitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  using namespace globals;

  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

               fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
               fputs("<!DOCTYPE Xdmf SYSTEM \"https://gitlab.kitware.com/xdmf/xdmf/raw/master/Xdmf.dtd\"[]>\n", xdmf_file_);
               fputs("<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2003/XInclude\">\n", xdmf_file_);
               fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", seedname.c_str());
               fputs("    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
               fflush(xdmf_file_);
}

//---------------------------------------------------------------------

void Hdf5Monitor::update_xdmf_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace H5;

  hsize_t      data_dimension  = 0;
  unsigned int float_precision = 8;

  if (float_type == PredType::IEEE_F32LE) {
    float_precision = 4;
  } else {
    float_precision = 8;
  }

  if (slice_.num_points() != 0) {
      data_dimension = static_cast<hsize_t>(slice_.num_points());
  } else {
      data_dimension = static_cast<hsize_t>(num_spins);
  }

               // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
               fseek(xdmf_file_, -31, SEEK_CUR);

  fprintf(xdmf_file_, "      <Grid Name=\"Lattice %d\" GridType=\"Uniform\">\n", solver->iteration());
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", solver->time());
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%llu\" />\n", data_dimension);
               fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s_lattice.h5:/positions\n", seedname.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Geometry>\n", xdmf_file_);
               fputs("       <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n", data_dimension);
  fprintf(xdmf_file_, "           %s_lattice.h5:/types\n", seedname.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Attribute>\n", xdmf_file_);
               fputs("       <Attribute Name=\"spin\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/spins\n", h5_file_name.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Attribute>\n", xdmf_file_);
   fputs("      </Grid>\n", xdmf_file_);

               // reprint the closing tags of the XML
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
