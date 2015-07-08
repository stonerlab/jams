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

#include "H5Cpp.h"

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

  config.lookupValue("sim.verbose_output", verbose_output_is_set);

  bool use_dipole = false;

  config.lookupValue("lattice.dipole", use_dipole);

  if (::optimize::use_fft) {
    //compute_fft_exchange_interactions();
    compute_exchange_interactions();
    if (use_dipole) {
      compute_fft_dipole_interactions();
    }
  } else {
    compute_exchange_interactions();
    if (use_dipole) {
      jams_error("Dipole calculations we requested but are unavailable due to the lack of FFT optimizations");
    }
  }
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
      lattice_vectors_[i][j] = lattice_settings["basis"][j][i];
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

  // if (!(lattice_pbc_.x && lattice_pbc_.y && lattice_pbc_.z)) {
  //   jams_warning("FFT optimizations are not yet supported for open boundaries.\nFFT OPTIMIZATIONS HAVE BEEN DISABLED");
  // }

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

  // if (counter > 1) {
  //   ::optimize::use_fft = false;
  //   jams_warning("FFT optimizations were requested,\nbut this is only supported with a single species.\nFFT OPTIMIZATIONS HAVE BEEN DISABLED");
  // }

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


  ::output.write("\ncalculating unit cell kpoint mesh...\n");
  calculate_unit_cell_kmesh();
  ::output.write("\nunit cell kpoints\n  %d  %d  %d\n", kpoints_.x, kpoints_.y, kpoints_.z);

  ::output.write("\nkspace size\n  %d  %d  %d\n", kpoints_.x*lattice_size_.x, kpoints_.y*lattice_size_.y, kpoints_.z*lattice_size_.z);
}

void Lattice::compute_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings) {

  fast_integer_lattice_.resize(lattice_size_.x, lattice_size_.y, lattice_size_.z, motif_.size());

  jblib::Vec3<int> kmesh_size(kpoints_.x*lattice_size_.x, kpoints_.y*lattice_size_.y, kpoints_.z*lattice_size_.z);
  if (!lattice_pbc_.x || !lattice_pbc_.y || !lattice_pbc_.z) {
    ::output.write("\nzero padding non-periodic dimensions\n");
     // double any non-periodic dimensions for zero padding
    for (int i = 0; i < 3; ++i) {
      if (!lattice_pbc_[i]) {
        kmesh_size[i] = 2*kpoints_[i]*lattice_size_[i];
      }
    }
    ::output.write("\npadded kspace size\n  %d  %d  %d\n", kmesh_size.x, kmesh_size.y, kmesh_size.z);
  }

  kspace_size_ = jblib::Vec3<int>(kmesh_size.x, kmesh_size.y, kmesh_size.z);
  kspace_map_.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z);

// initialize everything to -1 so we can check for double assignment below

  for (int i = 0, iend = lattice_size_.x*lattice_size_.y*lattice_size_.z*motif_.size(); i < iend; ++i) {
    fast_integer_lattice_[i] = -1;
  }

  for (int i = 0, iend = kspace_size_.x*kspace_size_.y*kspace_size_.z; i < iend; ++i) {
    kspace_map_[i] = -1;
  }

//-----------------------------------------------------------------------------
// Generate the realspace lattice positions
//-----------------------------------------------------------------------------

  int atom_counter = 0;
  rmax.x = 0.0; rmax.y = 0.0; rmax.z = 0.0;
  // loop over the translation vectors for lattice size
  for (int i = 0; i < lattice_size_.x; ++i) {
    for (int j = 0; j < lattice_size_.y; ++j) {
      for (int k = 0; k < lattice_size_.z; ++k) {
        // loop over atoms in the motif
        for (int m = 0, mend = motif_.size(); m != mend; ++m) {

          // number the site in the fast integer lattice
          fast_integer_lattice_(i, j, k, m) = atom_counter;

          // position of motif atom in fractional lattice vectors
          jblib::Vec3<double> lattice_pos(i+motif_[m].second.x, j+motif_[m].second.y, k+motif_[m].second.z);
          // position in real (cartesian) space
          jblib::Vec3<double> real_pos = lattice_vectors_*lattice_pos;

          // store max coordinates
          for (int n = 0; n < 3; ++n) {
            if (real_pos[n] > rmax[n]) {
              rmax[n] = real_pos[n];
            }
          }

          lattice_positions_.push_back(real_pos);
          lattice_materials_.push_back(motif_[m].first);
          lattice_material_num_.push_back(materials_map_[motif_[m].first]);

          jblib::Vec3<double> kvec((i+motif_[m].second.x)*kpoints_.x, (j+motif_[m].second.y)*kpoints_.y, (k+motif_[m].second.z)*kpoints_.z);

          // check that the motif*kpoints is comsurate (within a tolerance) to the integer kspace_lattice
          if (fabs(nint(kvec.x)-kvec.x) > 0.01 || fabs(nint(kvec.y)-kvec.y) > 0.01 || fabs(nint(kvec.z)-kvec.z) > 0.01) {
            jams_error("kpoint mesh does not map to the unit cell");
          }

          if (kspace_map_(nint(kvec.x), nint(kvec.y), nint(kvec.z)) != -1) {
            jams_error("attempted to assign multiple spins to the same point in the kspace map");
          }
          kspace_map_(nint(kvec.x), nint(kvec.y), nint(kvec.z)) = atom_counter;

          atom_counter++;
        }
      }
    }
  }

  if (atom_counter == 0) {
    jams_error("the number of computed lattice sites was zero, check input");
  }

  // std::ofstream kspacefile("kspace_map.dat");
  // for (int i = 0; i < kspace_size_.x; ++i) {
  //   for (int j = 0; j < kspace_size_.y; ++j) {
  //     for (int k = 0; k < kspace_size_.z; ++k) {
  //       kspacefile << i << "\t" << j << "\t" << k << "\t"<< kspace_map_(i,j,k) << std::endl;
  //     }
  //   }
  // }

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

  kspace_inv_map_.resize(globals::num_spins, 3);

  for (int i = 0; i < kspace_size_.x; ++i) {
    for (int j = 0; j < kspace_size_.y; ++j) {
      for (int k = 0; k < kspace_size_.z; ++k) {
        if (kspace_map_(i,j,k) != -1) {
          kspace_inv_map_(kspace_map_(i,j,k), 0) = i;
          kspace_inv_map_(kspace_map_(i,j,k), 1) = j;
          kspace_inv_map_(kspace_map_(i,j,k), 2) = k;
        }
      }
    }
  }



//-----------------------------------------------------------------------------
// initialize global arrays
//-----------------------------------------------------------------------------
  globals::s.resize(globals::num_spins, 3);

  // default spin array to (0, 0, 1) which will be used if no other spin settings
  // are specified
  for (int i = 0; i < globals::num_spins; ++i) {
    globals::s(i, 0) = 0.0;
    globals::s(i, 1) = 0.0;
    globals::s(i, 2) = 1.0;
  }

  // read initial spin config if specified
  if (lattice_settings.exists("spins")) {
    std::string spin_filename = lattice_settings["spins"];

    ::output.write("  reading initial spin configuration from: %s\n", spin_filename.c_str());

    load_spin_state_from_hdf5(spin_filename);
  }

  globals::h.resize(globals::num_spins, 3);
  globals::h_dipole.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);
  globals::wij.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z, 3, 3);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  std::fill(globals::h_dipole.data(), globals::h_dipole.data()+globals::num_spins3, 0.0);
  std::fill(globals::wij.data(), globals::wij.data()+kspace_size_.x*kspace_size_.y*kspace_size_.z*3*3, 0.0);

  material_count_.resize(num_materials(), 0);
  for (int i = 0; i < globals::num_spins; ++i) {
    int material_number = materials_map_[lattice_materials_[i]];
    material_count_[material_number]++;

    libconfig::Setting& type_settings = material_settings[material_number];

    // Setup the initial spin configuration if we haven't already read in a spin state
    if (!lattice_settings.exists("spins")) {
      if (type_settings["spin"].getType() == libconfig::Setting::TypeString) {
        // spin setting is a string
        std::string spin_initializer = capitalize(type_settings["spin"]);
        if (spin_initializer == "RANDOM") {
          rng.sphere(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
        } else {
          jams_error("Unknown spin initializer %s selected", spin_initializer.c_str());
        }
      } else if (type_settings["spin"].getType() == libconfig::Setting::TypeArray) {
        if (type_settings["spin"].getLength() == 3) {
          // spin setting is cartesian
          for(int j = 0; j < 3;++j) {
            globals::s(i, j) = type_settings["spin"][j];
          }
        } else if (type_settings["spin"].getLength() == 2) {
          // spin setting is spherical
          double theta = deg_to_rad(type_settings["spin"][0]);
          double phi   = deg_to_rad(type_settings["spin"][1]);

          globals::s(i, 0) = sin(theta)*cos(phi);
          globals::s(i, 1) = sin(theta)*sin(phi);
          globals::s(i, 2) = cos(theta);
        } else {
          jams_error("Spin initializer array must be 2 (spherical) or 3 (cartesian) components");
        }
      }
    }

    // normalise all spins
    double norm = sqrt(globals::s(i, 0)*globals::s(i, 0) + globals::s(i, 1)*globals::s(i, 1) + globals::s(i, 2)*globals::s(i, 2));
    for(int j = 0; j < 3;++j){
      globals::s(i, j) = globals::s(i, j)/norm;
    }

    // read material properties
    globals::mus(i) = type_settings["moment"];
    globals::alpha(i) = type_settings["alpha"];

    if (type_settings.exists("gyro")) {
      globals::gyro(i) = type_settings["gyro"];
    } else {
      // default
      globals::gyro(i) = 1.0;
    }

    globals::gyro(i) = -globals::gyro(i)/((1.0+globals::alpha(i)*globals::alpha(i))*globals::mus(i));
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

  fast_integer_interaction_list_.resize(motif_.size());

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
    jblib::Vec3<double> lattice_vector = (inverse_lattice_vectors_*interaction_vector) + motif_[typeA].second;

    // translate by the motif back to (hopefully) the origin of the local unit cell
    // for (int i = 0; i < 3; ++ i) {
    //   lattice_vector[i] -= motif_[type_difference].second[i];
    // }

    // this 4-vector specifies the integer number of lattice vectors to the unit cell and the fourth
    // component is the atoms number within the motif
    jblib::Vec4<int> fast_integer_vector;
    for (int i = 0; i < 3; ++ i) {
      // rounding with nint accounts for lack of precision in definition of the real space vectors
      fast_integer_vector[i] = floor(lattice_vector[i]+0.001);
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

    fast_integer_interaction_list_[typeA].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(fast_integer_vector, tensor));

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

void Lattice::load_spin_state_from_hdf5(std::string &filename) {
  using namespace H5;

  H5File file(filename.c_str(), H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet("/spins");
  DataSpace dataspace = dataset.getSpace();

  if (dataspace.getSimpleExtentNpoints() != static_cast<hssize_t>(globals::num_spins3)){
    jams_error("Spin state file '%s' has %llu spins but your simulation has %d spins.",
      filename.c_str(), dataspace.getSimpleExtentNpoints()/3, globals::num_spins);
  }

  dataset.read(globals::s.data(), PredType::NATIVE_DOUBLE);
}

void Lattice::compute_exchange_interactions() {

  globals::J1ij_t.resize(globals::num_spins3,globals::num_spins3);
  globals::J1ij_t.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
  globals::J1ij_t.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);

  ::output.write("\ncomputed interactions\n");

  bool is_all_inserts_successful = true;
  int counter = 0;
  // loop over the translation vectors for lattice size
  for (int i = 0; i < lattice_size_.x; ++i) {
    for (int j = 0; j < lattice_size_.y; ++j) {
      for (int k = 0; k < lattice_size_.z; ++k) {
        // loop over atoms in the motif
        for (int m = 0, mend = motif_.size(); m < mend; ++m) {
          int local_site = fast_integer_lattice_(i, j, k, m);

          std::vector<bool> is_already_interacting(globals::num_spins, false);
          is_already_interacting[local_site] = true;  // don't allow self interaction

          // loop over all possible interaction vectors
          for (int n = 0, nend = fast_integer_interaction_list_[m].size(); n < nend; ++n) {

            jblib::Vec4<int> fast_integer_lookup_vector(
              i + fast_integer_interaction_list_[m][n].first.x,
              j + fast_integer_interaction_list_[m][n].first.y,
              k + fast_integer_interaction_list_[m][n].first.z,
              (motif_.size() + m + fast_integer_interaction_list_[m][n].first.w)%motif_.size());

            bool interaction_is_outside_lattice = false;
            // if we are trying to interact with a site outside of the boundary
            for (int l = 0; l < 3; ++l) {
              if (lattice_pbc_[l]) {
                fast_integer_lookup_vector[l] = (fast_integer_lookup_vector[l] + lattice_size_[l])%lattice_size_[l];
              } else {
                if (fast_integer_lookup_vector[l] < 0 || fast_integer_lookup_vector[l] >= lattice_size_[l]) {
                  interaction_is_outside_lattice = true;
                }
              }
            }
            if (interaction_is_outside_lattice) {
              continue;
            }

            int neighbour_site = fast_integer_lattice_(fast_integer_lookup_vector.x, fast_integer_lookup_vector.y, fast_integer_lookup_vector.z, fast_integer_lookup_vector.w);

            // failsafe check that we only interact with any given site once through the input exchange file.
            if (is_already_interacting[neighbour_site]) {
              jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_site, fast_integer_lookup_vector.x, fast_integer_lookup_vector.y, fast_integer_lookup_vector.z, fast_integer_lookup_vector.w);
            }
            is_already_interacting[neighbour_site] = true;

            if (insert_interaction(local_site, neighbour_site, fast_integer_interaction_list_[m][n].second)) {
              // if(local_site >= neighbour_site) {
              //   std::cerr << local_site << "\t" << neighbour_site << "\t" << neighbour_site << "\t" << local_site << std::endl;
              // }
              counter++;
            } else {
              is_all_inserts_successful = false;
            }
            if (insert_interaction(neighbour_site, local_site, fast_integer_interaction_list_[m][n].second)) {
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

void Lattice::compute_fft_exchange_interactions() {
  ::output.write("\ncomputed fft exchange interactions\n");

  for (int i = 0, iend = motif_.size(); i < iend; ++i) {
    for (int j = 0, jend = fast_integer_interaction_list_[i].size(); j < jend; ++j) {
      jblib::Vec3<int> pos(
        kpoints_.x*(fast_integer_interaction_list_[i][j].first.x + motif_[fast_integer_interaction_list_[i][j].first.w].second.x),
        kpoints_.y*(fast_integer_interaction_list_[i][j].first.y + motif_[fast_integer_interaction_list_[i][j].first.w].second.y),
        kpoints_.z*(fast_integer_interaction_list_[i][j].first.z + motif_[fast_integer_interaction_list_[i][j].first.w].second.z));

      if (abs(pos.x) > (kspace_size_.x/2) || abs(pos.y) > (kspace_size_.y/2) || abs(pos.z) > (kspace_size_.z/2)) {
        jams_error("Your exchange is too long-range for the periodic system resulting in self interaction");
      }

      if (pos.x < 0) {
        pos.x = kspace_size_.x + pos.x;
      }
      if (pos.y < 0) {
        pos.y = kspace_size_.y + pos.y;
      }
      if (pos.z < 0) {
        pos.z = kspace_size_.z + pos.z;
      }

    // std::cerr <<fast_integer_interaction_list_[i].first.x << "\t" << fast_integer_interaction_list_[i].first.y << "\t" << fast_integer_interaction_list_[i].first.z << "\t" << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;

      for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 3; ++n) {
          globals::wij(pos.x, pos.y, pos.z, m, n) += fast_integer_interaction_list_[i][j].second[m][n]/mu_bohr_si;
        }
      }
    }
  }
}

void Lattice::compute_fft_dipole_interactions() {

  ::output.write("\ncomputed fft dipole interactions\n");

  jblib::Vec3<double> kmax = lattice_vectors_*jblib::Vec3<double>(kspace_size_.x/2, kspace_size_.y/2, kspace_size_.z/2);

  // only use rcut in periodic directions
  std::vector<double> kmax_pbc;
  for (int i = 0; i < 3; ++i) {
    if (lattice_pbc_[i]) {
      kmax_pbc.push_back(kmax[i]);
    }
  }
  double rcut = (*std::min_element(kmax_pbc.begin(),kmax_pbc.end()))*lattice_parameter_;

  ::output.write("\ndipole cutoff radius: %fnm\n", rcut);

  // loop over wij and calculate the rij parameters
  for (int i = 0; i < kspace_size_.x; ++i) {
    for (int j = 0; j < kspace_size_.y; ++j) {
      for (int k = 0; k < kspace_size_.z; ++k) {

        if (i == 0 && j == 0 && k == 0) {
          continue;
        }

        // position of motif atom in fractional lattice vectors
        jblib::Vec3<double> pos(i, j, k);

        if ( i > (kspace_size_.x/2) ) {
          pos.x = periodic_shift(pos.x, kspace_size_.x/2) - kspace_size_.x/2;
        }
        if ( j > (kspace_size_.y/2) ) {
          pos.y = periodic_shift(pos.y, kspace_size_.y/2) - kspace_size_.y/2;
        }
        if ( k > (kspace_size_.z/2) ) {
          pos.z = periodic_shift(pos.z, kspace_size_.z/2) - kspace_size_.z/2;
        }

        for (int m = 0; m < 3; ++m) {
          pos[m] = pos[m]/kpoints_[m];
        }

        jblib::Vec3<double> rij = lattice_vectors_*pos;

        rij *= lattice_parameter_;

        double rsq = 0.0;

        // only use a circular cut off for periodic directions
        // non-periodic directions us full distance
        for (int m = 0; m < 3; ++m) {
          if (lattice_pbc_[m]) {
            rsq += rij[m]*rij[m];
          }
        }

        if (rsq <= rcut*rcut) {

          //std::cerr << i << "\t" << j << "\t" << k << "\t" << rij.x << "\t" << rij.y << "\t" << rij.z << std::endl;

          jblib::Vec3<double> eij = rij/abs(rij);

          const double r = sqrt(dot(rij,rij));

          // identity matrix
          jblib::Matrix<double, 3, 3> ii( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

          for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
              globals::wij(i, j, k, m, n) += globals::mus(0)*globals::mus(0)*(mu_bohr_si*1E-7/(1E-27))*(3.0*eij[m]*eij[n]-ii[m][n])/(r*r*r);
            }
          }
          //std::cerr << i << "\t" << j << "\t" << k << "\t" << rij.x << "\t" << rij.y << "\t" << rij.z << "\t" << globals::wij(i, j, k, 2, 2) << std::endl;
        }
      }
    }
  }

}

bool Lattice::insert_interaction(const int m, const int n, const jblib::Matrix<double, 3, 3> &value) {

  int counter = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (fabs(value[i][j]) > energy_cutoff_) {
        counter++;
        if(globals::J1ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(globals::J1ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
            if(m >= n){
              globals::J1ij_t.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
            }
          }else{
            if(m <= n){

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

// Calculate the number of kpoints needed in each dimension to represent the unit cell
// i.e. the number of unique coordinates of each realspace dimension in the motif
void Lattice::calculate_unit_cell_kmesh() {

  const double tolerance = 0.001;

  std::vector<double> unique_x, unique_y, unique_z;

  unique_x.push_back(motif_[0].second.x);
  unique_y.push_back(motif_[0].second.y);
  unique_z.push_back(motif_[0].second.z);


  for (int i = 1, iend = motif_.size(); i < iend; ++i) {
    bool duplicate = false;
    for (int j = 0, jend = unique_x.size(); j < jend; ++j) {
      if (fabs(motif_[i].second.x - unique_x[j]) < tolerance) {
        duplicate = true;
        break;
      }
    }
    if(!duplicate) {
      unique_x.push_back(motif_[i].second.x);
    }
  }

  for (int i = 1, iend = motif_.size(); i < iend; ++i) {
    bool duplicate = false;
    for (int j = 0, jend = unique_y.size(); j < jend; ++j) {
      if (fabs(motif_[i].second.y - unique_y[j]) < tolerance) {
        duplicate = true;
        break;
      }
    }
    if(!duplicate) {
      unique_y.push_back(motif_[i].second.y);
    }
  }

  for (int i = 1, iend = motif_.size(); i < iend; ++i) {
    bool duplicate = false;
    for (int j = 0, jend = unique_z.size(); j < jend; ++j) {
      if (fabs(motif_[i].second.z - unique_z[j]) < tolerance) {
        duplicate = true;
        break;
      }
    }
    if(!duplicate) {
      unique_z.push_back(motif_[i].second.z);
    }
  }

  kpoints_.x = unique_x.size();
  kpoints_.y = unique_y.size();
  kpoints_.z = unique_z.size();
}

void Lattice::output_spin_state_as_vtu(std::ofstream &outfile){
  using namespace globals;

  outfile << "<?xml version=\"1.0\"?>" << "\n";
  outfile << "<VTKFile type=\"UnstructuredGrid\">" << "\n";
  outfile << "<UnstructuredGrid>" << "\n";
  outfile << "<Piece NumberOfPoints=\"" << num_spins << "\"  NumberOfCells=\"1\">" << "\n";
  outfile << "<PointData Scalar=\"Spins\">" << "\n";

  for(int n=0; n < materials_numbered_list_.size(); ++n){
    outfile << "<DataArray type=\"Float32\" Name=\"" << materials_numbered_list_[n] << "Spin\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
    for(int i = 0; i<num_spins; ++i){
      if(lattice_material_num_[i] == n){
        outfile << s(i, 0) << "\t" << s(i, 1) << "\t" << s(i, 2) << "\n";
      } else {
        outfile << 0.0 << "\t" << 0.0 << "\t" << 0.0 << "\n";
      }
    }
    outfile << "</DataArray>" << "\n";
  }

  outfile << "</PointData>" << "\n";
  outfile << "<CellData>" << "\n";
  outfile << "</CellData>" << "\n";
  outfile << "<Points>" << "\n";
  outfile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  for(int i = 0; i<num_spins; ++i){
    outfile << lattice_parameter_*lattice_positions_[i].x << "\t" << lattice_parameter_*lattice_positions_[i].y << "\t" << lattice_parameter_*lattice_positions_[i].z << "\n";
  }
  outfile << "</DataArray>" << "\n";
  outfile << "</Points>" << "\n";
  outfile << "<Cells>" << "\n";
  outfile << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "</Cells>" << "\n";
  outfile << "</Piece>" << "\n";
  outfile << "</UnstructuredGrid>" << "\n";
  outfile << "</VTKFile>" << "\n";

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

  if (filenum_spins != num_spins) {
    jams_error("I/O error, spin state file has %d spins but simulation has %d spins", filenum_spins, num_spins);
  } else {
    infile.read(reinterpret_cast<char*>(s.data()), num_spins3*sizeof(double));
  }

  if (infile.bad()) {
    jams_error("I/O error. Unknown failure reading spin state file");
  }
}
