//
// Created by Joseph Barker on 2019-05-02.
//

// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdio>
#include <climits>
#include <string>
#include <algorithm>
#include <vector>

#include "version.h"

#include "jams/helpers/error.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/slice.h"
#include "jams/interface/highfive.h"
#include "jams/containers/vec3.h"

#include "jams/monitors/unitcell_average.h"

using namespace std;

namespace {
    const unsigned h5_compression_chunk_size = 4095;
    const unsigned h5_compression_factor = 6;
}

UnitcellAverageMonitor::UnitcellAverageMonitor(const libconfig::Setting &settings)
    : Monitor(settings),
      slice_() {
  output_step_freq_ = settings["output_steps"];

  open_new_xdmf_file(jams::instance().output_path() + "/" + globals::simulation_name + "_avg.xdmf");

  spin_transformations_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    spin_transformations_[i] = globals::lattice->material(globals::lattice->atom_material_id(i)).transform;
  }

  cell_centers_.resize(globals::lattice->num_cells(), 3);

  for (auto i = 0; i < globals::lattice->num_cells(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      cell_centers_(i, j) = globals::lattice->cell_center(i)[j];
    }
  }

  cell_mag_.resize(globals::lattice->num_cells(), 3);
  cell_neel_.resize(globals::lattice->num_cells(), 3);
}

UnitcellAverageMonitor::~UnitcellAverageMonitor() {
  fclose(xdmf_file_);
}



void UnitcellAverageMonitor::update(Solver& solver) {
  int outcount = solver.iteration()/output_step_freq_;  // int divisible by modulo above

  const std::string h5_file_name(jams::instance().output_path() + "/" + globals::simulation_name + "_" + zero_pad_number(outcount) + "_avg.h5");

  cell_mag_.zero();
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto cell = globals::lattice->cell_containing_atom(i);
    for (auto j = 0; j < 3; ++j) {
      cell_mag_(cell, j) += globals::mus(i)*globals::s(i,j);
    }
  }

  cell_neel_.zero();
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto cell = globals::lattice->cell_containing_atom(i);
    auto s_transformed = spin_transformations_[i] * Vec3{{globals::s(i,0), globals::s(i,1), globals::s(i,2)}};
    for (auto j = 0; j < 3; ++j) {
      cell_neel_(cell, j) += globals::mus(i)*s_transformed[j];
    }
  }

  write_h5_file(h5_file_name, solver.iteration(), solver.time(), solver.physics()->temperature());
  update_xdmf_file(h5_file_name, solver.time());
}

void UnitcellAverageMonitor::write_h5_file(const std::string &h5_file_name, const int iteration, const double time, const double temperature) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  DataSetCreateProps props;

  if (compression_enabled_) {
    props.add(Chunking({std::min(h5_compression_chunk_size, globals::lattice->num_cells()), 1}));
    props.add(Shuffle());
    props.add(Deflate(h5_compression_factor));
  }

  auto pos_dataset = file.createDataSet<double>("/positions",  DataSpace({globals::lattice->num_cells(), 3}), props);

  pos_dataset.write(cell_centers_);


  auto mag_dataset = file.createDataSet<double>("/magnetisation",  DataSpace({globals::lattice->num_cells(), 3}), props);

  mag_dataset.createAttribute<int>("iteration", DataSpace::From(iteration));
  mag_dataset.createAttribute<double>("time", DataSpace::From(time));
  mag_dataset.createAttribute<double>("temperature", DataSpace::From(temperature));

  mag_dataset.write(cell_mag_);

  auto neel_dataset = file.createDataSet<double>("/neel",  DataSpace({globals::lattice->num_cells(), 3}), props);

  neel_dataset.createAttribute<int>("iteration", DataSpace::From(iteration));
  neel_dataset.createAttribute<double>("time", DataSpace::From(time));
  neel_dataset.createAttribute<double>("temperature", DataSpace::From(temperature));

  neel_dataset.write(cell_neel_);
}

//---------------------------------------------------------------------

void UnitcellAverageMonitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

  fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
  fputs("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\"[]>\n", xdmf_file_);
  fputs("<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n", xdmf_file_);
  fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", globals::simulation_name.c_str());
  fputs("    <Grid Name=\"Time\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}

//---------------------------------------------------------------------

void UnitcellAverageMonitor::update_xdmf_file(const std::string &h5_file_name, const double time) {
  hsize_t      data_dimension  = globals::lattice->num_cells();
  unsigned int float_precision = 8;

  // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
  fseek(xdmf_file_, -31, SEEK_CUR);

  fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file_);
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", time);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%llu\" />\n", data_dimension);
  fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/positions\n", h5_file_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Geometry>\n", xdmf_file_);
  fputs("       <Attribute Name=\"neel\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/neel\n", h5_file_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("       <Attribute Name=\"magnetisation\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/magnetisation\n", file_basename_no_extension(h5_file_name).c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("      </Grid>\n", xdmf_file_);
  // reprint the closing tags of the XML
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
