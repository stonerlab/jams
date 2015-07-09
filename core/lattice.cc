// Copyright 2014 Joseph Barker. All rights reserved.

extern "C"{
    #include "spglib/spglib.h"
}

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
#include <set>

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

  config.lookupValue("sim.verbose_output", verbose_output_is_set);

  read_lattice(::config.lookup("materials"), ::config.lookup("lattice"));
  calculate_unit_cell_symmetry();
  compute_positions(::config.lookup("materials"), ::config.lookup("lattice"));
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


  ::output.write("\ncalculating unit cell kpoint mesh...\n");
  calculate_unit_cell_kmesh();
  ::output.write("\nunit cell kpoints\n  %d  %d  %d\n", kpoints_.x, kpoints_.y, kpoints_.z);

  ::output.write("\nkspace size\n  %d  %d  %d\n", kpoints_.x*lattice_size_.x, kpoints_.y*lattice_size_.y, kpoints_.z*lattice_size_.z);
}

void Lattice::compute_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings) {

  lattice_integer_lookup_.resize(lattice_size_.x, lattice_size_.y, lattice_size_.z, motif_.size());

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
    lattice_integer_lookup_[i] = -1;
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
          lattice_integer_lookup_(i, j, k, m) = atom_counter;

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

          // if (kspace_map_(nint(kvec.x), nint(kvec.y), nint(kvec.z)) != -1) {
          //   jams_error("attempted to assign multiple spins to the same point in the kspace map");
          // }
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
  // globals::wij.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z, 3, 3);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  std::fill(globals::h_dipole.data(), globals::h_dipole.data()+globals::num_spins3, 0.0);
  // std::fill(globals::wij.data(), globals::wij.data()+kspace_size_.x*kspace_size_.y*kspace_size_.z*3*3, 0.0);

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

void Lattice::calculate_unit_cell_symmetry() {
  ::output.write("symmetry analysis\n");

  int i, j;
  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = lattice_vectors_[i][j];
    }
  }

  double (*spg_positions)[3] = new double[motif_.size()][3];

  for (i = 0; i < motif_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      spg_positions[i][j] = motif_[i].second[j];
    }
  }

  int (*spg_types) = new int[motif_.size()];

  for (i = 0; i < motif_.size(); ++i) {
    spg_types[i] = materials_map_[motif_[i].first];
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, motif_.size(), 1e-5);

  ::output.write("  International: %s (%d)\n", spglib_dataset_->international_symbol, spglib_dataset_->spacegroup_number );
  ::output.write("  Hall symbol:   %s\n", spglib_dataset_->hall_symbol );

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  ::output.write("  Point group:   %s\n", ptsymbol);
  ::output.write("  Transformation matrix:\n");
  for ( i = 0; i < 3; i++ ) {
      ::output.write("  %f %f %f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }
  ::output.write("  Wyckoff letters:\n");
  for ( i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      ::output.write("  %c ", wl[spglib_dataset_->wyckoffs[i]]);
  }
  ::output.write("\n");
  ::output.write("  Equivalent atoms:\n");
  for (i = 0; i < spglib_dataset_->n_atoms; i++) {
      ::output.write("  %d ", spglib_dataset_->equivalent_atoms[i]);
  }
  ::output.write("\n");
  ::output.verbose("  Symmetry operations:\n");

  for (int i = 0; i < spglib_dataset_->n_operations; i++) {
    ::output.verbose("--- %d ---\n", i + 1);
    for (j = 0; j < 3; j++) {
      ::output.verbose("%2d %2d %2d\n",
      spglib_dataset_->rotations[i][j][0],
      spglib_dataset_->rotations[i][j][1],
      spglib_dataset_->rotations[i][j][2]);
    }
    ::output.verbose("%f %f %f\n",
    spglib_dataset_->translations[i][0],
    spglib_dataset_->translations[i][1],
    spglib_dataset_->translations[i][2]);
  }

  int primitive_num_atoms = motif_.size();
  double primitive_lattice[3][3];

  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      primitive_lattice[i][j] = spg_lattice[i][j];
    }
  }

  double (*primitive_positions)[3] = new double[motif_.size()][3];

  for (i = 0; i < motif_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      primitive_positions[i][j] = spg_positions[i][j];
    }
  }

  int (*primitive_types) = new int[motif_.size()];

  for (i = 0; i < motif_.size(); ++i) {
    primitive_types[i] = spg_types[i];
  }

  primitive_num_atoms = spg_find_primitive(primitive_lattice, primitive_positions, primitive_types, motif_.size(), 1e-5);

  // spg_find_primitive returns 0 if the unit cell is already primitive
  if (primitive_num_atoms != 0) {
    ::output.write("\n");
    ::output.write("unit cell is not a primitive cell\n");
    ::output.write("\n");
    ::output.write("  primitive lattice vectors:\n");

    for (int i = 0; i < 3; ++i) {
      ::output.write("  % 3.6f % 3.6f % 3.6f\n",
        primitive_lattice[i][0], primitive_lattice[i][1], primitive_lattice[i][2]);
    }
    ::output.write("\n");
    ::output.write("  primitive motif positions:\n");

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      ::output.write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, materials_numbered_list_[primitive_types[i]].c_str(),
        primitive_positions[i][0], primitive_positions[i][1], primitive_positions[i][2]);
      counter++;
    }
  }

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


// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(jblib::Vec3<int>& pos) const {
    bool is_within_lattice = true;
    for (int l = 0; l < 3; ++l) {
      if (is_periodic(l)) {
        pos[l] = (pos[l] + lattice.num_unit_cells(l))%lattice.num_unit_cells(l);
      } else {
        if (pos[l] < 0 || pos[l] >= lattice.num_unit_cells(l)) {
          is_within_lattice = false;
        }
      }
    }
    return is_within_lattice;
}

// same as the Vec3 version but accepts a Vec4 where the last component is the motif
// position difference
bool Lattice::apply_boundary_conditions(jblib::Vec4<int>& pos) const {
  jblib::Vec3<int> pos3(pos.x, pos.y, pos.z);
  bool is_within_lattice = apply_boundary_conditions(pos3);
  if (is_within_lattice) {
    pos.x = pos3.x;
    pos.y = pos3.y;
    pos.z = pos3.z;
  }
  return is_within_lattice;
}
