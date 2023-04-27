//
// Created by Joe Barker on 2017/10/04.
//

#include <iomanip>

#include "version.h"
#include <jams/common.h>
#include "jams/core/globals.h"
#include "jams/core/interactions.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/core/hamiltonian.h"
#include "jams/monitors/cuda_spin_current.h"
#include "jams/hamiltonian/exchange.h"
#include "jams/interface/highfive.h"
#include "jams/containers/sparse_matrix_builder.h"
#include "jams/helpers/output.h"

CudaSpinCurrentMonitor::CudaSpinCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {

  assert(jams::instance().mode() == jams::Mode::GPU);

  jams_warning("This monitor automatically identifies the FIRST exchange hamiltonian\n"
               "in the config and assumes the exchange interaction is DIAGONAL AND ISOTROPIC");

  do_h5_output = jams::config_optional<bool>(settings, "h5", false);
  h5_output_steps = jams::config_optional<unsigned>(settings, "h5_output_steps", output_step_freq_);

  if (do_h5_output) {
    open_new_xdmf_file(jams::output::full_path_filename("js.xdmf"));
  }

  const auto& exchange_hamiltonian = find_hamiltonian<ExchangeHamiltonian>(::globals::solver->hamiltonians());

  jams::SparseMatrix<Vec3>::Builder sparse_matrix_builder(globals::num_spins, globals::num_spins);

  const auto& nbr_list = exchange_hamiltonian.neighbour_list();
  for (auto n = 0; n < nbr_list.size(); ++n) {
    auto i = nbr_list[n].first[0];
    auto j = nbr_list[n].first[1];
    auto Jij = nbr_list[n].second[0][0];
    sparse_matrix_builder.insert(i, j, globals::lattice->displacement(i, j) * Jij);
  }

  std::cout << "    dipole sparse matrix builder memory " << sparse_matrix_builder.memory() / kBytesToMegaBytes << "(MB)\n";
  std::cout << "    building CSR matrix\n";
  interaction_matrix_ = sparse_matrix_builder
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "    exchange sparse matrix memory (CSR): " << interaction_matrix_.memory() / kBytesToMegaBytes << " (MB)\n";

  spin_current_rx_z.resize(globals::num_spins);
  spin_current_ry_z.resize(globals::num_spins);
  spin_current_rz_z.resize(globals::num_spins);

  outfile.open(jams::output::full_path_filename("js.tsv"));

  outfile << "time\t";
  outfile << "js_z_rx\tjs_z_ry\tjs_z_rz" << std::endl;

  outfile.setf(std::ios::right);
}

void CudaSpinCurrentMonitor::update(Solver& solver) {
  Vec3 js_z = execute_cuda_spin_current_kernel(
          stream,
          globals::num_spins,
          globals::s.device_data(),
          globals::gyro.device_data(),
          globals::mus.device_data(),
          reinterpret_cast<const double*>(interaction_matrix_.val_device_data()),
          interaction_matrix_.row_device_data(),
          interaction_matrix_.col_device_data(),
          spin_current_rx_z.device_data(),
          spin_current_ry_z.device_data(),
          spin_current_rz_z.device_data()
  );

  // The calculation is in JAMS internal units. We want to convert to SI for
  // output. Js should have dimensions of m/s. In JAMS lengths are in terms
  // of the lattice constant and times are in picoseconds so we need to multiply
  // by the lattice constant and convert 1/ps to 1/s.
  const double units = globals::lattice->parameter() * 1e12;

  outfile << jams::fmt::sci << solver.time() << "\t";
  for (auto m = 0; m < 3; ++m) {
      outfile << jams::fmt::sci << units * js_z[m] << "\t";
  }
  outfile << "\n";

  if (do_h5_output && solver.iteration()%h5_output_steps == 0) {
    int outcount = solver.iteration()/h5_output_steps;  // int divisible by modulo above
    const std::string h5_file_name(jams::instance().output_path() + "/" + globals::simulation_name + "_" + zero_pad_number(outcount) + "_js.h5");
    write_spin_current_h5_file(h5_file_name, solver.iteration(), solver.time());
    update_xdmf_file(h5_file_name, solver.time());
  }
}

CudaSpinCurrentMonitor::~CudaSpinCurrentMonitor() {
  outfile.close();

  if (xdmf_file_ != nullptr) {
    fclose(xdmf_file_);
    xdmf_file_ = nullptr;
  }
}

void CudaSpinCurrentMonitor::write_spin_current_h5_file(const std::string &h5_file_name, const int iteration, const double time) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  DataSetCreateProps props;

  jams::MultiArray<double, 2> js(globals::num_spins, 3);

  for (auto i = 0; i < globals::num_spins; ++i) {
    js(i,0) = spin_current_rx_z(i);
    js(i,1) = spin_current_ry_z(i);
    js(i,2) = spin_current_rz_z(i);
  }

  auto dataset = file.createDataSet<double>("/spin_current",  DataSpace::From(js));

  dataset.createAttribute<int>("iteration", DataSpace::From(iteration));
  dataset.createAttribute<double>("time", DataSpace::From(time));
}

void CudaSpinCurrentMonitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  // create xdmf_file_
  xdmf_file_ = fopen(std::string(jams::instance().output_path() + "/" + xdmf_file_name).c_str(), "w");

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

void CudaSpinCurrentMonitor::update_xdmf_file(const std::string &h5_file_name, const double time) {
  unsigned int data_dimension  = globals::num_spins;
  unsigned int float_precision = 8;

  // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
  fseek(xdmf_file_, -31, SEEK_CUR);

  fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file_);
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", time/1e-12);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%llu\" />\n", globals::num_spins);
  fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s_lattice.h5:/positions\n", globals::simulation_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Geometry>\n", xdmf_file_);
  fputs("       <Attribute Name=\"Type\" AttributeType=\"Scalar\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n", data_dimension);
  fprintf(xdmf_file_, "           %s_lattice.h5:/types\n", globals::simulation_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("       <Attribute Name=\"spin_current\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/spin_current\n", file_basename_no_extension(h5_file_name).c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("      </Grid>\n", xdmf_file_);

  // reprint the closing tags of the XML
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
