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

namespace {
    const int h5_compression_chunk_size = 4095;
    const int h5_compression_factor = 6;
}

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings),
  slice_() {
    output_step_freq_ = settings["output_steps"];

    // it outsteps is 0 then use max int instead - output will only be generated in the
    // constructor and destructor
    if (output_step_freq_ == 0){
      output_step_freq_ = INT_MAX;
    }

    // compression options
    settings.lookupValue("compressed", compression_enabled_);
  std::cout << "  compressed " << compression_enabled_ << "\n";

    if (settings.exists("slice")) {
        slice_ = Slice(settings["slice"]);
    }

    write_ds_dt_ = jams::config_optional<bool>(settings, "ds_dt", write_ds_dt_);
    write_fields_ = jams::config_optional<bool>(settings, "fields", write_fields_);
    write_energies_ = jams::config_optional<bool>(settings, "energies", write_energies_);

    open_new_xdmf_file(jams::output::full_path_filename(".xdmf"));

    write_lattice_h5_file(jams::output::full_path_filename("lattice.h5"));
}

Hdf5Monitor::~Hdf5Monitor() {
  // always write final in double precision
    write_spin_h5_file(jams::output::full_path_filename("final.h5"));
    update_xdmf_file(jams::output::full_path_filename("final.h5"), globals::solver->time());

    fclose(xdmf_file_);
}



void Hdf5Monitor::update(Solver& solver) {
  if (solver.iteration()%output_step_freq_ == 0) {
    int outcount = solver.iteration()/output_step_freq_;  // int divisible by modulo above

    const std::string h5_file_name(jams::output::full_path_filename_series(".h5", outcount));

    write_spin_h5_file(h5_file_name);
    update_xdmf_file(h5_file_name, solver.time());
  }
}

void Hdf5Monitor::write_spin_h5_file(const std::string &h5_file_name) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  const bool slice_enabled = (slice_.num_points() != 0);
  const int num_output_points = slice_enabled ? slice_.num_points() : globals::num_spins;

  auto write_vector_dataset = [&](const jams::MultiArray<double, 2>& field, const std::string& data_path) {
    if (!slice_enabled) {
      write_vector_field(field, data_path, file);
      return;
    }

    jams::MultiArray<double, 2> sliced_field(num_output_points, 3);
    for (int i = 0; i < num_output_points; ++i) {
      for (int j = 0; j < 3; ++j) {
        sliced_field(i, j) = field(slice_.index(i), j);
      }
    }
    write_vector_field(sliced_field, data_path, file);
  };

  write_vector_dataset(globals::s, "/spins");

  if (write_ds_dt_) {
    write_vector_dataset(globals::ds_dt, "/ds_dt");
  }

  if (write_energies_) {
    jams::MultiArray<double, 1> energies(globals::num_spins);
    zero(energies);
    for (auto &hamiltonian : globals::solver->hamiltonians()) {
      hamiltonian->calculate_energies(globals::solver->time());
      for (auto i = 0; i < globals::num_spins; ++i)
      {
        energies(i) += hamiltonian->energy(i);
      }
    }
    if (!slice_enabled) {
      write_scalar_field(energies, "/energies", file);
    } else {
      jams::MultiArray<double, 1> sliced_energies(num_output_points);
      for (int i = 0; i < num_output_points; ++i) {
        sliced_energies(i) = energies(slice_.index(i));
      }
      write_scalar_field(sliced_energies, "/energies", file);
    }
  }

  if (write_fields_) {
    globals::solver->compute_fields();
    write_vector_dataset(globals::h, "/fields");
  }
}

//---------------------------------------------------------------------

void Hdf5Monitor::write_lattice_h5_file(const std::string &h5_file_name) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  const bool slice_enabled = (slice_.num_points() != 0);
  const int num_output_points = slice_enabled ? slice_.num_points() : globals::num_spins;

  jams::MultiArray<int, 1>    types;
  jams::MultiArray<double, 1> moments;
  jams::MultiArray<double, 2> positions;

  types.resize(num_output_points);
  moments.resize(num_output_points);
  positions.resize(num_output_points, 3);

  if (slice_enabled) {
    for (int i = 0; i < num_output_points; ++i) {
      types(i) = slice_.type(i);
      moments(i) = globals::mus(slice_.index(i));
      for (int j = 0; j < 3; ++j) {
        positions(i, j) = slice_.position(i, j) / 1e-9;
      }
    }
  } else {
    for (int i = 0; i < num_output_points; ++i) {
      types(i) = globals::lattice->lattice_site_material_id(i);
      moments(i) = globals::mus(i);
      for (int j = 0; j < 3; ++j) {
        positions(i, j) = globals::lattice->parameter() * globals::lattice->lattice_site_position_cart(i)[j] / 1e-9;
      }
    }
  }

  auto type_dataset = file.createDataSet<int>("/types",  DataSpace(num_output_points));
  type_dataset.write(types);
  auto moment_dataset = file.createDataSet<double>("/moments",  DataSpace(num_output_points));
  moment_dataset.write(moments);
  auto pos_dataset = file.createDataSet<double>("/positions",  DataSpace({size_t(num_output_points),3}));
  auto pos_units_attribute = pos_dataset.createAttribute<std::string>("units", DataSpace::From(std::string("nm")));
  pos_units_attribute.write(std::string("nm"));
  pos_dataset.write(positions);

}

//---------------------------------------------------------------------

void Hdf5Monitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

               fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
               fputs("<!DOCTYPE Xdmf SYSTEM \"https://gitlab.kitware.com/xdmf/xdmf/raw/master/Xdmf.dtd\"[]>\n", xdmf_file_);
               fputs("<Xdmf Version=\"3.0\">\n", xdmf_file_);
               fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", globals::simulation_name.c_str());
               fputs("    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
               fflush(xdmf_file_);
}

//---------------------------------------------------------------------

void Hdf5Monitor::update_xdmf_file(const std::string &h5_file_name, const double time) {
  unsigned      data_dimension  = 0;
  unsigned int float_precision = 8;

  if (slice_.num_points() != 0) {
      data_dimension = static_cast<unsigned>(slice_.num_points());
  } else {
      data_dimension = static_cast<unsigned>(globals::num_spins);
  }

               // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
               fseek(xdmf_file_, -31, SEEK_CUR);

  fprintf(xdmf_file_, "      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n");
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", time);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%u\" />\n", data_dimension);
               fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s_lattice.h5:/positions\n", globals::simulation_name.c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Geometry>\n", xdmf_file_);
               fputs("       <Attribute Name=\"type\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n", data_dimension);
  fprintf(xdmf_file_, "           %s_lattice.h5:/types\n", globals::simulation_name.c_str());
                fputs("         </DataItem>\n", xdmf_file_);
                fputs("       </Attribute>\n", xdmf_file_);
                fputs("       <Attribute Name=\"moment\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s_lattice.h5:/moments\n", globals::simulation_name.c_str());
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
  if (write_fields_) {
    fputs("       <Attribute Name=\"fields\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
    fprintf(xdmf_file_,"         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
    fprintf(xdmf_file_, "           %s:/fields\n", file_basename(h5_file_name).c_str());
    fputs("         </DataItem>\n", xdmf_file_);
    fputs("       </Attribute>\n", xdmf_file_);
  }
  if (write_energies_) {
    fputs("       <Attribute Name=\"energies\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
    fprintf(xdmf_file_,"         <DataItem Dimensions=\"%u 1\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
    fprintf(xdmf_file_, "           %s:/energies\n", file_basename(h5_file_name).c_str());
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
                                     const std::string &data_path,
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
                                     const std::string &data_path,
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
