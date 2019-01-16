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
#include "jams/cuda/cuda_defs.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/monitors/cuda_spin_current.h"
#include "cuda_spin_current.h"
#include "jblib/containers/array.h"

using namespace std;

CudaSpinCurrentMonitor::CudaSpinCurrentMonitor(const libconfig::Setting &settings)
        : Monitor(settings) {

  assert(solver->is_cuda_solver());

  jams_warning("This monitor automatically identifies the FIRST exchange hamiltonian in the config");
  jams_warning("This monitor currently assumes the exchange interaction is DIAGONAL AND ISOTROPIC");

  do_h5_output = jams::config_optional<bool>(settings, "h5", false);
  h5_output_steps = jams::config_optional<unsigned>(settings, "h5_output_steps", output_step_freq_);

  if (do_h5_output) {
    open_new_xdmf_file(seedname + "_js.xdmf");
  }

  const auto& exchange_settings = config_find_setting_by_key_value_pair(config->lookup("hamiltonians"), "module", "exchange");

  const std::string exchange_file_name = exchange_settings["exc_file"];

  if (exchange_file_name.empty()) {
    throw std::runtime_error("no exchange hamiltonian found");
  }

  std::ifstream interaction_file(exchange_file_name);

  if (interaction_file.fail()) {
    throw std::runtime_error("failed to open interaction file:" + exchange_file_name);
  }

  cout << "    interaction file name: " << exchange_file_name << endl;

  const auto neighbour_list = generate_neighbour_list_from_file(exchange_settings, interaction_file);

  SparseMatrix<Vec3> interaction_matrix(globals::num_spins, globals::num_spins);

  for (auto i = 0; i < neighbour_list.size(); ++i) {
    for (auto const &nbr: neighbour_list[i]) {
      auto j = nbr.first;
      auto Jij = nbr.second[0][0];
      if (i > j) continue;
      auto r_i = lattice->atom_position(i);
      auto r_j = lattice->atom_position(j);
      interaction_matrix.insertValue(i, j, lattice->displacement(r_i, r_j) * Jij);
    }
  }

  cout << "  converting interaction matrix format from MAP to CSR\n";
  interaction_matrix.convertMAP2CSR();
  cout << "  exchange matrix memory (CSR): " << interaction_matrix.calculateMemory() << " MB\n";

  cuda_malloc_and_copy_to_device(dev_csr_matrix_.row, interaction_matrix.rowPtr(), interaction_matrix.rows()+1);
  cuda_malloc_and_copy_to_device(dev_csr_matrix_.col, interaction_matrix.colPtr(), interaction_matrix.nonZero());

  // not sure how Vec3 will copy so lets be safe
  jblib::Array<double, 2> val(interaction_matrix.nonZero(), 3);
  for (unsigned i = 0; i < interaction_matrix.nonZero(); ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      val(i, j) = interaction_matrix.val(i)[j];
    }
  }

  cuda_malloc_and_copy_to_device(dev_csr_matrix_.val, val.data(), val.elements());

  dev_spin_current_rx_x.resize(globals::num_spins);
  dev_spin_current_rx_y.resize(globals::num_spins);
  dev_spin_current_rx_z.resize(globals::num_spins);

  dev_spin_current_ry_x.resize(globals::num_spins);
  dev_spin_current_ry_y.resize(globals::num_spins);
  dev_spin_current_ry_z.resize(globals::num_spins);

  dev_spin_current_rz_x.resize(globals::num_spins);
  dev_spin_current_rz_y.resize(globals::num_spins);
  dev_spin_current_rz_z.resize(globals::num_spins);

  outfile.open(seedname + "_js.tsv");

  outfile << "time\t";
  outfile << "js_rx_x\tjs_rx_y\tjs_rx_z" << "\t";
  outfile << "js_ry_x\tjs_ry_y\tjs_yx_z" << "\t";
  outfile << "js_rz_x\tjs_rz_y\tjs_zx_z" << std::endl;

  outfile.setf(std::ios::right);
}

void CudaSpinCurrentMonitor::update(Solver *solver) {
  Mat3 js = execute_cuda_spin_current_kernel(
          stream,
          globals::num_spins,
          solver->dev_ptr_spin(),
          dev_csr_matrix_.val,
          dev_csr_matrix_.row,
          dev_csr_matrix_.col,
          dev_spin_current_rx_x.data(),
          dev_spin_current_rx_y.data(),
          dev_spin_current_rx_z.data(),
          dev_spin_current_ry_x.data(),
          dev_spin_current_ry_y.data(),
          dev_spin_current_ry_z.data(),
          dev_spin_current_rz_x.data(),
          dev_spin_current_rz_y.data(),
          dev_spin_current_rz_z.data()
  );

//  const double units = (lattice->parameter() * 1e-9) * kBohrMagneton * kGyromagneticRatio;

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
  fclose(xdmf_file_);

  if (dev_csr_matrix_.row) {
    cudaFree(dev_csr_matrix_.row);
    dev_csr_matrix_.row = nullptr;
  }

  if (dev_csr_matrix_.col) {
    cudaFree(dev_csr_matrix_.col);
    dev_csr_matrix_.col = nullptr;
  }

  if (dev_csr_matrix_.val) {
    cudaFree(dev_csr_matrix_.val);
    dev_csr_matrix_.val = nullptr;
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

  jblib::Array<double, 1> js_rx_z(num_spins);
  jblib::Array<double, 1> js_ry_z(num_spins);
  jblib::Array<double, 1> js_rz_z(num_spins);

  dev_spin_current_rx_z.copy_to_host_array(js_rx_z);
  dev_spin_current_ry_z.copy_to_host_array(js_ry_z);
  dev_spin_current_rz_z.copy_to_host_array(js_rz_z);

  jblib::Array<double, 2> js(num_spins, 3);

  for (auto i = 0; i < num_spins; ++i) {
    js(i,0) = js_rx_z(i);
    js(i,1) = js_ry_z(i);
    js(i,2) = js_rz_z(i);
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
