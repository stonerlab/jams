// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

#include "jams/core/output.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/utils.h"


#include "jams/monitors/unitcell-magnetisation.h"

#include "jblib/containers/vec.h"

UnitcellMagnetisationMonitor::UnitcellMagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  mag(lattice->size(0), lattice->size(1), lattice->size(2), 3)
{
  using namespace globals;
  ::output->write("\ninitialising Unitcell Magnetisation monitor\n");

  new_xdmf_file(seedname + "_unitcell_mag.xdmf");

}

void UnitcellMagnetisationMonitor::update(Solver * solver) {
  using namespace globals;
  using namespace H5;

  if (solver->iteration()%output_step_freq_ == 0) {
    mag.zero();
    for (int i = 0; i < lattice->size(0); ++i) {
      for (int j = 0; j < lattice->size(1); ++j) {
        for (int k = 0; k < lattice->size(2); ++k) {
          for (int n = 0; n < lattice->num_unit_cell_positions(); ++n) {
            const int index = lattice->site_index_by_unit_cell(i, j, k, n);
            for (int dim = 0; dim < 3; ++dim) {
              mag(i, j, k, dim) += globals::s(index, dim) * globals::mus(index);
            }
          }
        }
      }
    }

    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    const std::string h5_file_name(seedname + "_unitcell_" + zero_pad_number(outcount) + ".h5");

    write_h5_file(h5_file_name, PredType::NATIVE_DOUBLE);
    update_xdmf_file(h5_file_name, PredType::NATIVE_DOUBLE);
  }
}

void UnitcellMagnetisationMonitor::write_h5_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace H5;

  hsize_t dims[4];

  H5File outfile(h5_file_name.c_str(), H5F_ACC_TRUNC);
  
  dims[0] = lattice->size(0);
  dims[1] = lattice->size(1);
  dims[2] = lattice->size(2);
  dims[3] = 3;

  DataSpace dataspace(4, dims);

  DSetCreatPropList plist;

  double out_iteration = solver->iteration();
  double out_time = solver->time();
  double out_temperature = solver->physics()->temperature();
  jblib::Vec3<double> out_field = solver->physics()->applied_field();

  DataSet mag_dataset = outfile.createDataSet("magnetisation", float_type, dataspace, plist);

  DataSpace attribute_dataspace(H5S_SCALAR);
  Attribute attribute = mag_dataset.createAttribute("iteration", PredType::STD_I32LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_INT32, &out_iteration);
  attribute = mag_dataset.createAttribute("time", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_time);
  attribute = mag_dataset.createAttribute("temperature", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_temperature);
  attribute = mag_dataset.createAttribute("hx", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_field.x);
  attribute = mag_dataset.createAttribute("hy", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_field.y);
  attribute = mag_dataset.createAttribute("hz", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_field.z);

  mag_dataset.write(mag.data(), PredType::NATIVE_DOUBLE);

  outfile.close();
}

void UnitcellMagnetisationMonitor::new_xdmf_file(const std::string &xdmf_file_name) {
  using namespace globals;

  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

               fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
               fputs("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\"[]>\n", xdmf_file_);
               fputs("<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n", xdmf_file_);
               fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", seedname.c_str());
               fputs("    <Grid Name=\"Time\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
               fflush(xdmf_file_);
}

void UnitcellMagnetisationMonitor::update_xdmf_file(const std::string &h5_file_name, const H5::PredType float_type) {
  using namespace globals;
  using namespace H5;

  hsize_t      data_dimension  = 0;
  unsigned int float_precision = 8;

  
               // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
               fseek(xdmf_file_, -31, SEEK_CUR);

               fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file_);
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", solver->time()/1e-12);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"3DRectMesh\" Dimensions=\"%u %u %u\"/>\n", lattice->size(0), lattice->size(1), lattice->size(2));
               fputs("       <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n", xdmf_file_);
               fputs("         <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n", xdmf_file_);
               fputs("           0 0 0\n", xdmf_file_);
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("         <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\"\n>", xdmf_file_);
  fprintf(xdmf_file_, "           %f %f %f\n", 1.0, 1.0, 1.0);
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Geometry>\n", xdmf_file_);
               fputs("       <Attribute Name=\"magnetisation\" AttributeType=\"Vector\" Center=\"Cell\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu %llu %llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", lattice->size(0), lattice->size(1), lattice->size(2), float_precision);
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

UnitcellMagnetisationMonitor::~UnitcellMagnetisationMonitor() {
  fclose(xdmf_file_);
}
