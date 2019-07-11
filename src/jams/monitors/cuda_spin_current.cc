//
// Created by Joe Barker on 2017/10/04.
//

#include <iomanip>
#include "H5Cpp.h"

#include "version.h"
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
#include "cuda_spin_current.h"
#include "jams/hamiltonian/exchange.h"

using namespace std;

CudaSpinCurrentMonitor::CudaSpinCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {

  assert(solver->is_cuda_solver());

  jams_warning("This monitor automatically identifies the FIRST exchange hamiltonian\n"
               "in the config and assumes the exchange interaction is DIAGONAL AND ISOTROPIC");

  do_h5_output = jams::config_optional<bool>(settings, "h5", false);
  h5_output_steps = jams::config_optional<unsigned>(settings, "h5_output_steps", output_step_freq_);

  if (do_h5_output) {
    open_new_xdmf_file(seedname + "_js.xdmf");
  }

  const auto exchange_hamiltonian = find_hamiltonian<ExchangeHamiltonian>(::solver->hamiltonians());
  assert (exchange_hamiltonian != nullptr);

  SparseMatrix<Vec3> interaction_matrix(globals::num_spins, globals::num_spins);

  for (auto i = 0; i < exchange_hamiltonian->neighbour_list().size(); ++i) {
    for (auto const &nbr: exchange_hamiltonian->neighbour_list()[i]) {
      auto j = nbr.first;
      auto Jij = nbr.second[0][0];
      auto r_i = lattice->atom_position(i);
      auto r_j = lattice->atom_position(j);
      interaction_matrix.insertValue(i, j, lattice->displacement(i, j) * Jij);
    }
  }

  cout << "  converting interaction matrix format from MAP to CSR\n";
  interaction_matrix.convertMAP2CSR();
  cout << "  exchange matrix memory (CSR): " << interaction_matrix.calculateMemory() << " MB\n";

  dev_csr_matrix_.row.resize(interaction_matrix.rows()+1);
  std::copy(interaction_matrix.rowPtr(), interaction_matrix.rowPtr()+interaction_matrix.rows()+1, dev_csr_matrix_.row.data());

  dev_csr_matrix_.col.resize(interaction_matrix.nonZero());
  std::copy(interaction_matrix.colPtr(), interaction_matrix.colPtr()+interaction_matrix.nonZero(), dev_csr_matrix_.col.data());


  // not sure how Vec3 will copy so lets be safe
  dev_csr_matrix_.val.resize(3*interaction_matrix.nonZero());
  int count = 0;
  for (unsigned i = 0; i < interaction_matrix.nonZero(); ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      dev_csr_matrix_.val(count) = interaction_matrix.val(i)[j];
      count++;
    }
  }

  spin_current_rx_x.resize(globals::num_spins);
  spin_current_rx_y.resize(globals::num_spins);
  spin_current_rx_z.resize(globals::num_spins);

  spin_current_ry_x.resize(globals::num_spins);
  spin_current_ry_y.resize(globals::num_spins);
  spin_current_ry_z.resize(globals::num_spins);

  spin_current_rz_x.resize(globals::num_spins);
  spin_current_rz_y.resize(globals::num_spins);
  spin_current_rz_z.resize(globals::num_spins);

  outfile.open(seedname + "_js.tsv");

  outfile << "time\t";
  outfile << "js_rx_x\tjs_rx_y\tjs_rx_z" << "\t";
  outfile << "js_ry_x\tjs_ry_y\tjs_ry_z" << "\t";
  outfile << "js_rz_x\tjs_rz_y\tjs_rz_z" << std::endl;

  outfile.setf(std::ios::right);
}

void CudaSpinCurrentMonitor::update(Solver *solver) {
  Mat3 js = execute_cuda_spin_current_kernel(
          stream,
          globals::num_spins,
          globals::s.device_data(),
          dev_csr_matrix_.val.device_data(),
          dev_csr_matrix_.row.device_data(),
          dev_csr_matrix_.col.device_data(),
          spin_current_rx_x.device_data(),
          spin_current_rx_y.device_data(),
          spin_current_rx_z.device_data(),
          spin_current_ry_x.device_data(),
          spin_current_ry_y.device_data(),
          spin_current_ry_z.device_data(),
          spin_current_rz_x.device_data(),
          spin_current_rz_y.device_data(),
          spin_current_rz_z.device_data()
  );

//  const double units = lattice->parameter() * kBohrMagneton * kGyromagneticRatio;

  outfile << std::setw(4) << std::scientific << solver->time() << "\t";
  for (auto r_m = 0; r_m < 3; ++r_m) {
    for (auto s_n = 0; s_n < 3; ++ s_n) {
      outfile << std::setw(12) << js[r_m][s_n] << "\t";
    }
  }
  outfile << "\n";

  if (do_h5_output && solver->iteration()%h5_output_steps == 0) {
    int outcount = solver->iteration()/h5_output_steps;  // int divisible by modulo above
    const std::string h5_file_name(seedname + "_" + zero_pad_number(outcount) + "_js.h5");
    write_spin_current_h5_file(h5_file_name);
    update_xdmf_file(h5_file_name);
  }
}

bool CudaSpinCurrentMonitor::is_converged() {
  return false;
}

CudaSpinCurrentMonitor::~CudaSpinCurrentMonitor() {
  outfile.close();

  if (xdmf_file_ != nullptr) {
    fclose(xdmf_file_);
    xdmf_file_ = nullptr;
  }
}

void CudaSpinCurrentMonitor::write_spin_current_h5_file(const std::string &h5_file_name) {
  using namespace globals;
  using namespace H5;

  hsize_t dims[2];

  dims[0] = static_cast<hsize_t>(num_spins);
  dims[1] = 3;

  H5File outfile(h5_file_name.c_str(), H5F_ACC_TRUNC);

  DataSpace dataspace(2, dims);

  DSetCreatPropList plist;

  double out_iteration = solver->iteration();
  double out_time = solver->time();

  DataSet spin_dataset = outfile.createDataSet("spin_current", PredType::NATIVE_DOUBLE, dataspace, plist);

  DataSpace attribute_dataspace(H5S_SCALAR);
  Attribute attribute = spin_dataset.createAttribute("iteration", PredType::STD_I32LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_INT32, &out_iteration);
  attribute = spin_dataset.createAttribute("time", PredType::IEEE_F64LE, attribute_dataspace);
  attribute.write(PredType::NATIVE_DOUBLE, &out_time);

  jams::MultiArray<double, 2> js(num_spins, 3);

  for (auto i = 0; i < num_spins; ++i) {
    js(i,0) = spin_current_rx_z(i);
    js(i,1) = spin_current_ry_z(i);
    js(i,2) = spin_current_rz_z(i);
  }

  spin_dataset.write(js.data(), PredType::NATIVE_DOUBLE);
}

void CudaSpinCurrentMonitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
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

void CudaSpinCurrentMonitor::update_xdmf_file(const std::string &h5_file_name) {
  using namespace globals;
  using namespace H5;

  hsize_t      data_dimension  = static_cast<hsize_t>(num_spins);
  unsigned int float_precision = 8;

  // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
  fseek(xdmf_file_, -31, SEEK_CUR);

  fputs("      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n", xdmf_file_);
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", solver->time()/1e-12);
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
  fputs("       <Attribute Name=\"spin_current\" AttributeType=\"Vector\" Center=\"Node\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, float_precision);
  fprintf(xdmf_file_, "           %s:/spin_current\n", h5_file_name.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
  fputs("      </Grid>\n", xdmf_file_);

  // reprint the closing tags of the XML
  fputs("    </Grid>\n", xdmf_file_);
  fputs("  </Domain>\n", xdmf_file_);
  fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
