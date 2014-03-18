// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/lattice.h"

#include <libconfig.h++>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "core/consts.h"
#include "core/globals.h"
#include "core/maths.h"
#include "core/sparsematrix.h"
#include "core/utils.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"


void Lattice::initialize() {
  ::output.write("\n----------------------------------------\n");
  ::output.write("initializing lattice");
  ::output.write("\n----------------------------------------\n");

  read_lattice(::config.lookup("materials"), ::config.lookup("lattice"));
  compute_positions(::config.lookup("materials"), ::config.lookup("lattice"));
  read_interactions(::config.lookup("lattice"));
  compute_interactions();
}

///
/// @brief  Read lattice settings from config file.
///
void Lattice::read_lattice(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings) {
  using namespace globals;

//-----------------------------------------------------------------------------
// Read lattice vectors
//-----------------------------------------------------------------------------

  // We transpose during the read because the unit cell matrix must have the
  // lattice vectors as the columns but it is easiest to define each vector in
  // the input
  // | a1x a2x a2x |  | A |   | A.a1x + B.a2x + C.a3x |
  // | a1y a2y a3y |  | B | = | A.a1y + B.a2y + C.a3y |
  // | a1z a2z a3z |  | C |   | A.a1z + B.a2z + C.a3z |
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      lattice_vectors_[i][j] = lattice_settings["basis"][i][j];
    }
  }
  ::output.write("\nlattice translation vectors\n");
  for (int i = 0; i < 3; ++i) {
    ::output.write("  % 3.6f % 3.6f % 3.6f\n",
      lattice_vectors_[i][0], lattice_vectors_[i][1], lattice_vectors_[i][2]);
  }

  inverse_lattice_vectors_ = lattice_vectors_.inverse();
  ::output.write("\ninverse lattice vectors\n");
  for (int i = 0; i < 3; ++i) {
    ::output.write("  % 3.6f % 3.6f % 3.6f\n",
      inverse_lattice_vectors_[i][0], inverse_lattice_vectors_[i][1], inverse_lattice_vectors_[i][2]);
  }

  lattice_parameter_ = lattice_settings["parameter"];
  if (lattice_parameter_ < 0.0) {
    jams_error("lattice parameter cannot be negative");
  }
  ::output.write("\nlattice parameter (nm)\n  %3.6f\n", lattice_parameter_);

  for (int i = 0; i < 3; ++i) {
    lattice_size_[i] = lattice_settings["size"][i];
  }
  ::output.write("\nlattice size\n  %d  %d  %d\n",
    lattice_size_.x, lattice_size_.y, lattice_size_.z);

  // TODO: read kpoints

//-----------------------------------------------------------------------------
// Read boundary conditions
//-----------------------------------------------------------------------------

  if(!lattice_settings.exists("periodic")) {
    // sane default
    ::output.write(
      "\nNo boundary conditions specified - assuming 3D periodic\n");
    for (int i = 0; i < 3; ++i) {
      lattice_pbc_[i] = true;
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      lattice_pbc_[i] = lattice_settings["periodic"][i];
    }
  }
  ::output.write("\nboundary conditions\n  %s  %s  %s\n",
    lattice_pbc_.x ? "periodic" : "open",
    lattice_pbc_.y ? "periodic" : "open",
    lattice_pbc_.z ? "periodic" : "open");

//-----------------------------------------------------------------------------
// Read materials
//-----------------------------------------------------------------------------

  ::output.write("\nmaterials\n");

  int counter = 0;
  for (int i = 0; i < material_settings.getLength(); ++i) {
    std::string name = material_settings[i]["name"];
    if (materials_map_.insert( std::pair<std::string, int>(name, counter)).second == false) {
      jams_error("the material %s is specified twice in the configuration", name.c_str());
    }
    materials_numbered_list_.push_back(name);
    ::output.write("  %-6d %s\n", counter, name.c_str());
    counter++;
  }

//-----------------------------------------------------------------------------
// Read motif
//-----------------------------------------------------------------------------

  // TODO - use libconfig to check if this is a string or a group to allow
  // positions to be defined in the config file directly
  std::string position_filename = lattice_settings["positions"];

  std::ifstream position_file(position_filename.c_str());

  if(position_file.fail()) {
    jams_error("failed to open position file %s", position_filename.c_str());
  }

  ::output.write("\nlattice motif (%s)\n", position_filename.c_str());

  counter = 0;
  // read the motif into an array from the positions file
  for (std::string line; getline(position_file, line); ) {
    std::stringstream is(line);
    std::pair<std::string, jblib::Vec3<double> > atom;
    // read atom type name
    is >> atom.first;

    // check the material type is defined
    if (materials_map_.find(atom.first) == materials_map_.end()) {
      jams_error("material %s in the motif is not defined in the configuration", atom.first.c_str());
    }

    // read atom coordinates
    is >> atom.second.x >> atom.second.y >> atom.second.z;

    motif_.push_back(atom);
    ::output.write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, atom.first.c_str(), atom.second.x, atom.second.y, atom.second.z);
    counter++;
  }
  position_file.close();
}

void Lattice::compute_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings) {

  fast_integer_lattice_.resize(
    lattice_size_.x, lattice_size_.y, lattice_size_.z, motif_.size());

//-----------------------------------------------------------------------------
// Generate the realspace lattice positions
//-----------------------------------------------------------------------------

  int atom_counter = 0;
  // loop over the translation vectors for lattice size
  for (int i = 0; i != lattice_size_.x; ++i) {
    for (int j = 0; j != lattice_size_.y; ++j) {
      for (int k = 0; k != lattice_size_.z; ++k) {
        // loop over atoms in the motif
        for (int m = 0, mend = motif_.size(); m != mend; ++m) {

          // number the site in the fast integer lattice
          fast_integer_lattice_(i, j, k, m) = atom_counter;

          // position of motif atom in fractional lattice vectors
          jblib::Vec3<double> lattice_pos(
            i+motif_[m].second.x, j+motif_[m].second.y, k+motif_[m].second.z);
          // position in real (cartesian) space
          jblib::Vec3<double> real_pos = lattice_vectors_*lattice_pos;

          lattice_positions_.push_back(real_pos);
          lattice_materials_.push_back(motif_[m].first);

          atom_counter++;
        }
      }
    }
  }

  if (atom_counter == 0) {
    jams_error("the number of computed lattice sites was zero, check input");
  }

  globals::num_spins = atom_counter;
  globals::num_spins3 = 3*atom_counter;

  ::output.write("\ncomputed lattice positions\n");
  if (verbose_output_is_set) {
    for (int i = 0, iend = lattice_positions_.size(); i != iend; ++i) {
      ::output.write("  %-6d %-6s % 3.6f % 3.6f % 3.6f\n", i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z);
    }
  } else {
    // avoid spamming the screen by default
    for (int i = 0; i != 8; ++i) {
    ::output.write("  %-6d %-6s %3.6f % 3.6f % 3.6f\n", i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z);
  }
    if (lattice_positions_.size() > 0) {
      ::output.write("  ... [use verbose output for details] ... \n");
    }
  }
  ::output.write("  total: %d\n", atom_counter);

//-----------------------------------------------------------------------------
// initialize global arrays
//-----------------------------------------------------------------------------
  globals::s.resize(globals::num_spins, 3);
  globals::h.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);
  globals::d2z.resize(globals::num_spins);
  globals::d4z.resize(globals::num_spins);
  globals::d6z.resize(globals::num_spins);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  std::fill(globals::d2z.data(), globals::d2z.data()+globals::num_spins, 0.0);
  std::fill(globals::d4z.data(), globals::d4z.data()+globals::num_spins, 0.0);
  std::fill(globals::d6z.data(), globals::d6z.data()+globals::num_spins, 0.0);

  material_count_.resize(num_materials(), 0);
  for (int i = 0; i != globals::num_spins; ++i) {
    int material_number = materials_map_[lattice_materials_[i]];
    material_count_[material_number]++;

    libconfig::Setting& type_settings = material_settings[material_number];

    bool randomize_spins_is_set = false;
    type_settings.lookupValue("spin_random",randomize_spins_is_set);

    // read initial spin state
    if(randomize_spins_is_set){
      rng.sphere(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    }else{
      for(int j = 0; j != 3;++j) {
        globals::s(i, j) = type_settings["spin"][j];
      }
    }

    // normalise spin
    double norm = sqrt(globals::s(i, 0)*globals::s(i, 0) + globals::s(i, 1)*globals::s(i, 1) + globals::s(i, 2)*globals::s(i, 2));
    for(int j = 0; j != 3;++j){
      globals::s(i, j) = globals::s(i, j)/norm;
    }

    // read material properties
    globals::mus(i) = type_settings["moment"];
    globals::alpha(i) = type_settings["alpha"];
    globals::gyro(i) = type_settings["gyro"];
    globals::gyro(i) = -globals::gyro(i)/((1.0+globals::alpha(i)*globals::alpha(i))*globals::mus(i));

    if ( (type_settings.exists("K1") || type_settings.exists("K2") || type_settings.exists("K3")) &&
         (type_settings.exists("d2z") || type_settings.exists("d4z") || type_settings.exists("d6z")) ) {
      jams_error("anisotropy should only be specified in terms of K1, K2, K3 or d2z, d4z, d6z in the config file");
    }

    if (type_settings.exists("d2z")) {
      globals::d2z(i) = type_settings["d2z"];
      globals::d2z(i) /= mu_bohr_si;
    }

    if (type_settings.exists("d4z")) {
      globals::d4z(i) = type_settings["d4z"];
      globals::d4z(i) /= mu_bohr_si;
    }

    if (type_settings.exists("d6z")) {
      globals::d6z(i) = type_settings["d6z"];
      globals::d6z(i) /= mu_bohr_si;
    }

    double K1 = 0.0, K2 = 0.0, K3 = 0.0;
    if (type_settings.exists("K3")) {
      K3 = type_settings["K3"];
      globals::d6z(i) = -((16.0/231.0)*K3)/mu_bohr_si;
    }

    if (type_settings.exists("K2")) {
      K2 = type_settings["K2"];
      globals::d4z(i) = ((8.0/35.0)*K2 + (144.0/385.0)*K3)/mu_bohr_si;
    }

    if (type_settings.exists("K1")) {
      K1 = type_settings["K1"];
      globals::d2z(i) = -(2.0/3.0)*(K1 + (8.0/7.0)*K2 + (8.0/7.0)*K3)/mu_bohr_si;
    }
  }
}

void Lattice::read_interactions(const libconfig::Setting &lattice_settings) {
  if (!lattice_settings.exists("exchange")) {
    jams_warning("No exchange interaction file specified");
    return; // don't try and process the file
  }

  std::string interaction_filename = lattice_settings["exchange"];
  // read in typeA typeB rx ry rz Jij
  std::ifstream interaction_file(interaction_filename.c_str());

  if(interaction_file.fail()) {
    jams_error("failed to open interaction file %s", interaction_filename.c_str());
  }

  ::output.write("\ninteraction vectors (%s)\n", interaction_filename.c_str());


  int counter = 0;
  // read the motif into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {
    std::stringstream is(line);

    int typeA, typeB;

    is >> typeA >> typeB;
    typeA--; typeB--;  // zero base the types

    // type difference
    int type_difference = (typeB - typeA);

    jblib::Vec3<double> interaction_vector;
    is >> interaction_vector.x >> interaction_vector.y >> interaction_vector.z;

    // transform into lattice vector basis
    jblib::Vec3<double> lattice_vector = (inverse_lattice_vectors_*interaction_vector);

    // translate by the motif back to (hopefully) the origin of the local unit cell
    for (int i = 0; i < 3; ++ i) {
      lattice_vector[i] -= motif_[type_difference].second[i];
    }

    // this 4-vector specifies the integer number of lattice vectors to the unit cell and the fourth
    // component is the atoms number within the motif
    jblib::Vec4<int> fast_integer_vector;
    for (int i = 0; i < 3; ++ i) {
      // rounding with nint accounts for lack of precision in definition of the real space vectors
      fast_integer_vector[i] = nint(lattice_vector[i]);
    }
    fast_integer_vector[3] = type_difference;

    jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (file_columns(line) == 6) {
      // one Jij component given - diagonal
      is >> tensor[0][0];
      tensor[1][1] = tensor[0][0];
      tensor[2][2] = tensor[0][0];
    } else if (file_columns(line) == 14) {
      // nine Jij components given - full tensor
      is >> tensor[0][0] >> tensor[0][1] >> tensor[0][2];
      is >> tensor[1][0] >> tensor[1][1] >> tensor[1][2];
      is >> tensor[2][0] >> tensor[2][1] >> tensor[2][2];
    } else {
      jams_error("number of Jij values in exchange files must be 1 or 9, check your input on line %d", counter);
    }

    fast_integer_interaction_list_.push_back(
      std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(
        fast_integer_vector, tensor) );

    if (verbose_output_is_set) {
      ::output.write("  line %-9d              %s\n", counter, line.c_str());
      ::output.write("  types               A : %d  B : %d  B - A : %d\n", typeA, typeB, type_difference);
      ::output.write("  interaction vector % 3.6f % 3.6f % 3.6f\n",
        interaction_vector.x, interaction_vector.y, interaction_vector.z);
      ::output.write("  fractional vector  % 3.6f % 3.6f % 3.6f\n",
        lattice_vector.x, lattice_vector.y, lattice_vector.z);
      ::output.write("  integer vector     % -9d % -9d % -9d % -9d\n\n",
        fast_integer_vector.x, fast_integer_vector.y, fast_integer_vector.z, fast_integer_vector.w);
    }
    counter++;
  }

  if(!verbose_output_is_set) {
    ::output.write("  ... [use verbose output for details] ... \n");
    ::output.write("  total: %d\n", counter);
  }

  interaction_file.close();

  energy_cutoff_ = 1E-26;  // default value (in joules)
  lattice_settings.lookupValue("energy_cutoff", energy_cutoff_);

  ::output.write("\ninteraction energy cutoff\n  %e\n", energy_cutoff_);
}

void Lattice::compute_interactions() {

  globals::J1ij_t.resize(globals::num_spins3,globals::num_spins3);

  ::output.write("\ncomputed interactions\n");

  bool is_all_inserts_successful = true;
  int counter = 0;
  // loop over the translation vectors for lattice size
  for (int i = 0; i != lattice_size_.x; ++i) {
    for (int j = 0; j != lattice_size_.y; ++j) {
      for (int k = 0; k != lattice_size_.z; ++k) {
        // loop over atoms in the motif
        for (int m = 0, mend = motif_.size(); m != mend; ++m) {
          std::vector<bool> is_already_interacting(globals::num_spins, false);

          int local_site = fast_integer_lattice_(i, j, k, m);

          is_already_interacting[local_site] = true;  // don't allow self interaction
          // loop over all possible interaction vectors
          for (int n = 0, nend = fast_integer_interaction_list_.size(); n != nend; ++n) {

            jblib::Vec4<int> fast_integer_lookup_vector(
              i + fast_integer_interaction_list_[n].first.x,
              j + fast_integer_interaction_list_[n].first.y,
              k + fast_integer_interaction_list_[n].first.z,
              (motif_.size() + m + fast_integer_interaction_list_[n].first.w)%motif_.size());

            for (int l = 0; l < 3; ++l) {
              if (lattice_pbc_[l]) {
                fast_integer_lookup_vector[l] = (fast_integer_lookup_vector[l] + lattice_size_[l])%lattice_size_[l];
              } else {
                // TODO: check the interaction is within the system bounds
                jams_error("open boundaries are not implmented");
              }
            }

            int neighbour_site = fast_integer_lattice_(
              fast_integer_lookup_vector.x, fast_integer_lookup_vector.y,
              fast_integer_lookup_vector.z, fast_integer_lookup_vector.w);

            // failsafe check that we only interact with any given site once through the input exchange file.
            if (is_already_interacting[neighbour_site]) {
              jams_error("Multiple interactions between spins %d and %d. Check the exchange file.", local_site, neighbour_site);
            }
            is_already_interacting[neighbour_site] = true;

            if (insert_interaction(local_site, neighbour_site, fast_integer_interaction_list_[n].second)) {
              counter++;
            } else {
              is_all_inserts_successful = false;
            }
          }
        }
      }
    }
  }

  if (!is_all_inserts_successful) {
    jams_warning("Some interactions were ignored due to the energy cutoff (%e)", energy_cutoff_);
  }

  ::output.write("  total: %d\n", counter);
}

bool Lattice::insert_interaction(const int m, const int n, const jblib::Matrix<double, 3, 3> &value) {

  int counter = 0;
  for (int i = 0; i != 3; ++i) {
    for (int j = 0; j != 3; ++j) {
      if (fabs(value[i][j]) > energy_cutoff_) {
        counter++;
        if(globals::J1ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(globals::J1ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
            if(i >= j){
              globals::J1ij_t.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
            }
          }else{
            if(i <= j){
              globals::J1ij_t.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
            }
          }
        }else{
          globals::J1ij_t.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
        }
      }
    }
  }

  if (counter == 0) {
    return false;
  }

  return true;
}



void Lattice::output_spin_state_as_vtu(std::ofstream &outfile){
  // using namespace globals;

  // outfile << "<?xml version=\"1.0\"?>" << "\n";
  // outfile << "<VTKFile type=\"UnstructuredGrid\">" << "\n";
  // outfile << "<UnstructuredGrid>" << "\n";
  // outfile << "<Piece NumberOfPoints=\""<<num_spins<<"\"  NumberOfCells=\"1\">" << "\n";
  // outfile << "<PointData Scalar=\"Spins\">" << "\n";

  // for(int n=0; n < num_types(); ++n){
  //   outfile << "<DataArray type=\"Float32\" Name=\"" << atom_names[n] << "Spin\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  //   for(int i = 0; i<num_spins; ++i){
  //     if(atom_type[i] == n){
  //       outfile << s(i, 0) << "\t" << s(i, 1) << "\t" << s(i, 2) << "\n";
  //     } else {
  //       outfile << 0.0 << "\t" << 0.0 << "\t" << 0.0 << "\n";
  //     }
  //   }
  //   outfile << "</DataArray>" << "\n";
  // }
  // outfile << "</PointData>" << "\n";
  // outfile << "<CellData>" << "\n";
  // outfile << "</CellData>" << "\n";
  // outfile << "<Points>" << "\n";
  // outfile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  // for(int i = 0; i<num_spins; ++i){
  //   outfile << atom_pos(i, 0) << "\t" << atom_pos(i, 1) << "\t" << atom_pos(i, 2) << "\n";
  // }
  // outfile << "</DataArray>" << "\n";
  // outfile << "</Points>" << "\n";
  // outfile << "<Cells>" << "\n";
  // outfile << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << "\n";
  // outfile << "1" << "\n";
  // outfile << "</DataArray>" << "\n";
  // outfile << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << "\n";
  // outfile << "1" << "\n";
  // outfile << "</DataArray>" << "\n";
  // outfile << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << "\n";
  // outfile << "1" << "\n";
  // outfile << "</DataArray>" << "\n";
  // outfile << "</Cells>" << "\n";
  // outfile << "</Piece>" << "\n";
  // outfile << "</UnstructuredGrid>" << "\n";
  // outfile << "</VTKFile>" << "\n";

}

void Lattice::output_spin_state_as_binary(std::ofstream &outfile){
  using namespace globals;

  outfile.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
  outfile.write(reinterpret_cast<char*>(s.data()), num_spins3*sizeof(double));
}

void Lattice::output_spin_types_as_binary(std::ofstream &outfile){
  using namespace globals;

  // outfile.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
  // outfile.write(reinterpret_cast<char*>(&atom_type[0]), num_spins*sizeof(int));
}

void Lattice::read_spin_state_from_binary(std::ifstream &infile){
  using namespace globals;

  infile.seekg(0);

  int filenum_spins=0;
  infile.read(reinterpret_cast<char*>(&filenum_spins), sizeof(int));

  if(filenum_spins != num_spins){
    jams_error("I/O error, spin state file has %d spins but simulation has %d spins", filenum_spins, num_spins);
  }else{
    infile.read(reinterpret_cast<char*>(s.data()), num_spins3*sizeof(double));
  }

  if(infile.bad()){
    jams_error("I/O error. Unknown failure reading spin state file");
  }
}
