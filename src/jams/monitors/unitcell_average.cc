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
#include "H5Cpp.h"

#include "jams/helpers/error.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/slice.h"
#include "jams/interface/h5.h"
#include "jams/containers/vec3.h"

#include "jams/monitors/unitcell_average.h"

using namespace std;

namespace {
    const unsigned h5_compression_chunk_size = 4095;
    const unsigned h5_compression_factor = 6;
}

UnitcellAverageMonitor::UnitcellAverageMonitor(const libconfig::Setting &settings)
    : Monitor(settings),
      float_pred_type_(H5::PredType::IEEE_F64LE),
      slice_() {
  using namespace globals;
  using namespace H5;

  output_step_freq_ = settings["output_steps"];

  open_new_xdmf_file(seedname + "_avg.xdmf");

  spin_transformations_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    spin_transformations_[i] = lattice->material(lattice->atom_material_id(i)).transform;
  }

  cell_centers_.resize(lattice->num_cells(), 3);

  for (auto i = 0; i < lattice->num_cells(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      cell_centers_(i, j) = lattice->cell_center(i)[j];
    }
  }

  cell_mag_.resize(lattice->num_cells(), 3);
  cell_neel_.resize(lattice->num_cells(), 3);
}

UnitcellAverageMonitor::~UnitcellAverageMonitor() {
  fclose(xdmf_file_);
}



void UnitcellAverageMonitor::update(Solver * solver) {
  using namespace globals;
  using namespace H5;

  int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

  const std::string h5_file_name(seedname + "_" + zero_pad_number(outcount) + "_avg.h5");

  cell_mag_.zero();
  for (auto i = 0; i < num_spins; ++i) {
    auto cell = lattice->cell_containing_atom(i);
    for (auto j = 0; j < 3; ++j) {
      cell_mag_(cell, j) += mus(i)*s(i,j);
    }
  }

  cell_neel_.zero();
  for (auto i = 0; i < num_spins; ++i) {
    auto cell = lattice->cell_containing_atom(i);
    auto s_transformed = spin_transformations_[i] * Vec3{{s(i,0), s(i,1), s(i,2)}};
    for (auto j = 0; j < 3; ++j) {
      cell_neel_(cell, j) += mus(i)*s_transformed[j];
    }
  }

  write_h5_file(h5_file_name, float_pred_type_);
  update_xdmf_file(h5_file_name, float_pred_type_);
}

void UnitcellAverageMonitor::write_h5_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace HighFive;


  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  DataSetCreateProps props;

  if (compression_enabled_) {
    props.add(Chunking({std::min(h5_compression_chunk_size, lattice->num_cells()), 1}));
    props.add(Shuffle());
    props.add(Deflate(h5_compression_factor));
  }

  auto pos_dataset = file.createDataSet<double>("/positions",  DataSpace({lattice->num_cells(), 3}), props);

  pos_dataset.write(cell_centers_);


  auto mag_dataset = file.createDataSet<double>("/magnetisation",  DataSpace({lattice->num_cells(), 3}), props);

  mag_dataset.createAttribute("iteration", solver->iteration());
  mag_dataset.createAttribute("time", solver->time());
  mag_dataset.createAttribute("temperature", solver->physics()->temperature());

  mag_dataset.write(cell_mag_);

  auto neel_dataset = file.createDataSet<double>("/neel",  DataSpace({lattice->num_cells(), 3}), props);

  neel_dataset.createAttribute("iteration", solver->iteration());
  neel_dataset.createAttribute("time", solver->time());
  neel_dataset.createAttribute("temperature", solver->physics()->temperature());

  neel_dataset.write(cell_neel_);
}

//---------------------------------------------------------------------

void UnitcellAverageMonitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  using namespace globals;

  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

  fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
  fputs("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\"[]>\n", xdmf_file_);
  fputs("<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n", xdmf_file_);
  fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", seedname.c_str());
  fputs("    <Grid Name=\"Time\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}

//---------------------------------------------------------------------

void UnitcellAverageMonitor::update_xdmf_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace H5;

  hsize_t      data_dimension  = lattice->num_cells();
  unsigned int float_precision = 8;

  // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
  fseek(xdmf_file_, -31, SEEK_CUR);

  fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file_);
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", solver->time());
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
  fprintf(xdmf_file_, "           %s:/magnetisation\n", h5_file_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("      </Grid>\n", xdmf_file_);
  // reprint the closing tags of the XML
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
