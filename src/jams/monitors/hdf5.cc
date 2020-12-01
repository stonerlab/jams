// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdio>
#include <climits>
#include <string>
#include <algorithm>

#include "version.h"

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/slice.h"
#include "jams/interface/highfive.h"
#include "jams/helpers/output.h"

#include "jams/monitors/hdf5.h"

using namespace std;

namespace {
    const int h5_compression_chunk_size = 4095;
    const int h5_compression_factor = 6;
}

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings),
  slice_() {
    using namespace globals;

    output_step_freq_ = settings["output_steps"];

    // it outsteps is 0 then use max int instead - output will only be generated in the
    // constructor and destructor
    if (output_step_freq_ == 0){
      output_step_freq_ = INT_MAX;
    }

    // compression options
    settings.lookupValue("compressed", compression_enabled_);
    cout << "  compressed " << compression_enabled_ << "\n";

    if (settings.exists("slice")) {
        slice_ = Slice(settings["slice"]);
    }

    write_ds_dt_ = jams::config_optional<bool>(settings, "ds_dt", write_ds_dt_);

    open_new_xdmf_file(jams::output::full_path_filename(".xdmf"));

    write_lattice_h5_file(jams::output::full_path_filename("lattice.h5"));
}

Hdf5Monitor::~Hdf5Monitor() {
  // always write final in double precision
    write_spin_h5_file(jams::output::full_path_filename("final.h5"));
    update_xdmf_file(jams::output::full_path_filename("final.h5"));

    fclose(xdmf_file_);
}



void Hdf5Monitor::update(Solver * solver) {
  using namespace globals;

  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    const std::string h5_file_name(jams::output::full_path_filename_series(".h5", outcount));

    write_spin_h5_file(h5_file_name);
    update_xdmf_file(h5_file_name);
  }
}

void Hdf5Monitor::write_spin_h5_file(const std::string &h5_file_name) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  write_vector_field(globals::s, "/spins", file);

  if (write_ds_dt_) {
    write_vector_field(globals::ds_dt, "/ds_dt", file);
  }
}

//---------------------------------------------------------------------

void Hdf5Monitor::write_lattice_h5_file(const std::string &h5_file_name) {
  using namespace globals;
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  jams::MultiArray<int, 1>    types;
  jams::MultiArray<double, 2> positions;

  if (slice_.num_points() != 0) {
    for (auto i = 0; i < slice_.num_points(); ++i) {
      types(i) = slice_.type(i);
    }

    positions.resize(slice_.num_points(), 3);

    for (auto i = 0; i < slice_.num_points(); ++i) {
      for (auto j = 0; j < 3; ++j) {
        positions(i, j) = slice_.position(i, j);
      }
    }
  } else {
    types.resize(num_spins);

    for (auto i = 0; i < num_spins; ++i) {
      types(i) = lattice->atom_material_id(i);
    }

    positions.resize(num_spins, 3);

    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        positions(i, j) = lattice->parameter() * lattice->atom_position(i)[j] / 1e-9;
      }
    }
  }

  auto type_dataset = file.createDataSet<int>("/types",  DataSpace(num_spins));
  type_dataset.write(types);
  auto pos_dataset = file.createDataSet<double>("/positions",  DataSpace({size_t(num_spins),3}));
  pos_dataset.write(positions);
}

//---------------------------------------------------------------------

void Hdf5Monitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  using namespace globals;

  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

               fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
               fputs("<!DOCTYPE Xdmf SYSTEM \"https://gitlab.kitware.com/xdmf/xdmf/raw/master/Xdmf.dtd\"[]>\n", xdmf_file_);
               fputs("<Xdmf Version=\"3.0\">\n", xdmf_file_);
               fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", simulation_name.c_str());
               fputs("    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
               fflush(xdmf_file_);
}

//---------------------------------------------------------------------

void Hdf5Monitor::update_xdmf_file(const std::string &h5_file_name) {
  using namespace globals;

  unsigned      data_dimension  = 0;
  unsigned int float_precision = 8;

  if (slice_.num_points() != 0) {
      data_dimension = static_cast<unsigned>(slice_.num_points());
  } else {
      data_dimension = static_cast<unsigned>(num_spins);
  }

               // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
               fseek(xdmf_file_, -31, SEEK_CUR);

  fprintf(xdmf_file_, "      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n");
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", solver->time()/1e-12);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%u\" />\n", data_dimension);
               fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s_lattice.h5:/positions\n", simulation_name.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Geometry>\n", xdmf_file_);
               fputs("       <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n", data_dimension);
  fprintf(xdmf_file_, "           %s_lattice.h5:/types\n", simulation_name.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Attribute>\n", xdmf_file_);
               fputs("       <Attribute Name=\"spin\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/spins\n", file_basename(h5_file_name).c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Attribute>\n", xdmf_file_);
               if (write_ds_dt_) {
                 fputs("       <Attribute Name=\"ds_dt\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
                 fprintf(xdmf_file_,"         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
                 fprintf(xdmf_file_, "           %s:/ds_dt\n", file_basename(h5_file_name).c_str());
                 fputs("         </DataItem>\n", xdmf_file_);
                 fputs("       </Attribute>\n", xdmf_file_);
               }
               fputs("      </Grid>\n", xdmf_file_);
               // reprint the closing tags of the XML
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}

void Hdf5Monitor::write_vector_field(const jams::MultiArray<double, 2> &field,
                                     const string &data_path,
                                     HighFive::File &file) const {
  using namespace HighFive;

  DataSetCreateProps props;

  if (compression_enabled_) {
    props.add(Chunking({static_cast<unsigned long long>(std::min(h5_compression_chunk_size, int(field.size(0)))), 1}));
    props.add(Shuffle());
    props.add(Deflate(h5_compression_factor));
  }

  auto dataset = file.createDataSet<double>(data_path,  DataSpace({size_t(field.size(0)), size_t(field.size(1))}), props);
  dataset.write(field);
}

void Hdf5Monitor::write_scalar_field(const jams::MultiArray<double, 1> &field,
                                     const string &data_path,
                                     HighFive::File &file) const {
  using namespace HighFive;

  DataSetCreateProps props;

  if (compression_enabled_) {
    props.add(Chunking({static_cast<unsigned long long>(std::min(h5_compression_chunk_size, int(field.size())))}));
    props.add(Shuffle());
    props.add(Deflate(h5_compression_factor));
  }

  auto dataset = file.createDataSet<double>(data_path,  DataSpace({size_t(field.size())}), props);
  dataset.write(field);
}
