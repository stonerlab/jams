// Copyright 2014 Joseph Barker. All rights reserved.

extern "C"{
    #include "spglib/spglib.h"
}

#include "core/lattice.h"

#include <libconfig.h++>
#include <stdint.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <set>
#include <cfloat>

#include "H5Cpp.h"

#include "core/consts.h"
#include "core/globals.h"
#include "core/maths.h"
#include "core/sparsematrix.h"
#include "core/utils.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

using std::cout;
using std::endl;

Lattice::Lattice() {

}

Lattice::~Lattice() {
  if (spglib_dataset_ != NULL) {
    spg_free_dataset(spglib_dataset_);
  }
}

jblib::Vec3<double> Lattice::generate_position(
  const jblib::Vec3<double> unit_cell_frac_pos,
  const jblib::Vec3<int> translation_vector) const
{
  return unit_cell_*generate_fractional_position(unit_cell_frac_pos, translation_vector);
}

// generate a position within a periodic image of the entire system
jblib::Vec3<double> Lattice::generate_image_position(
  const jblib::Vec3<double> unit_cell_cart_pos,
  const jblib::Vec3<int> image_vector) const
{
  jblib::Vec3<double> frac_pos = cartesian_to_fractional(unit_cell_cart_pos);
  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      frac_pos[n] = frac_pos[n] + image_vector[n]*lattice_size_[n];
    }
  }
  return fractional_to_cartesian(frac_pos);
}

jblib::Vec3<double> Lattice::generate_fractional_position(
  const jblib::Vec3<double> unit_cell_frac_pos,
  const jblib::Vec3<int> translation_vector) const
{
  return jblib::Vec3<double>(unit_cell_frac_pos.x + translation_vector.x,
                             unit_cell_frac_pos.y + translation_vector.y,
                             unit_cell_frac_pos.z + translation_vector.z);
}

void Lattice::init_from_config(const libconfig::Config& pConfig) {
  init_unit_cell(pConfig.lookup("materials"), pConfig.lookup("lattice"));
  init_lattice_positions(pConfig.lookup("materials"), pConfig.lookup("lattice"));
}

jblib::Vec3<double> Lattice::minimum_image(const jblib::Vec3<double> ri, const jblib::Vec3<double> rj) const {

  if(is_open_system()) {
    return rj - ri;
  }

  jblib::Vec3<double> dr = cartesian_to_fractional(rj - ri);

  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      dr[n] = dr[n] - nint(dr[n] / lattice_size_[n]) * lattice_size_[n];
    }
  }

  return fractional_to_cartesian(dr);
};

jblib::Vec3<double> Lattice::minimum_image_fractional(const jblib::Vec3<double> fi, const jblib::Vec3<double> fj) const {
  if(is_open_system()) {
    return fj - fi;
  }

  jblib::Vec3<double> dr = fj - fi;

  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      dr[n] = dr[n] - nint(dr[n] / lattice_size_[n]) * lattice_size_[n];
    }
  }
  return dr;
};

double Lattice::distance(const int i, const int j) const {
  return fractional_to_cartesian(
    minimum_image_fractional(lattice_frac_positions_[i], lattice_frac_positions_[j])
  ).norm();
}

double Lattice::distance_sq(const int i, const int j) const {
  return fractional_to_cartesian(
    minimum_image_fractional(lattice_frac_positions_[i], lattice_frac_positions_[j])
  ).norm_sq();
}

void Lattice::init_unit_cell(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings) {
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
      unit_cell_[i][j] = lattice_settings["basis"][i][j];
    }
  }
  printf("\n----------------------------------------\n");
  printf("\nunit cell\n");

  printf("  lattice vectors\n");
  for (int i = 0; i < 3; ++i) {
    printf("    % 3.6f % 3.6f % 3.6f\n",
      unit_cell_[i][0], unit_cell_[i][1], unit_cell_[i][2]);
  }

  unit_cell_inverse_ = unit_cell_.inverse();

  printf("  inverse lattice vectors\n");
  for (int i = 0; i < 3; ++i) {
    printf("    % 3.6f % 3.6f % 3.6f\n",
      unit_cell_inverse_[i][0], unit_cell_inverse_[i][1], unit_cell_inverse_[i][2]);
  }

  lattice_parameter_ = lattice_settings["parameter"];
  printf("  lattice parameter (m):\n    %3.6e\n", lattice_parameter_);

  if (lattice_parameter_ < 0.0) {
    throw std::runtime_error("config: lattice parameter cannot be negative");
  }

  printf("  unitcell volume (m^3):\n    %3.6e\n", this->volume());

  for (int i = 0; i < 3; ++i) {
    lattice_size_[i] = lattice_settings["size"][i];
  }

  printf("  lattice size\n    %d  %d  %d\n",
    lattice_size_.x, lattice_size_.y, lattice_size_.z);

//-----------------------------------------------------------------------------
// Read boundary conditions
//-----------------------------------------------------------------------------

  if(!lattice_settings.exists("periodic")) {
    // sane default
    printf(
      "\nASSUMPTION: no boundary conditions specified - assuming 3D periodic\n");
    for (int i = 0; i < 3; ++i) {
      lattice_periodic_boundary_[i] = true;
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      lattice_periodic_boundary_[i] = lattice_settings["periodic"][i];
    }
  }
  printf("  boundary conditions\n    %s  %s  %s\n",
    lattice_periodic_boundary_.x ? "periodic" : "open",
    lattice_periodic_boundary_.y ? "periodic" : "open",
    lattice_periodic_boundary_.z ? "periodic" : "open");

//-----------------------------------------------------------------------------
// Read materials
//-----------------------------------------------------------------------------

  printf("  materials\n");

  int counter = 0;
  for (int i = 0; i < material_settings.getLength(); ++i) {
    std::string name = material_settings[i]["name"];
    if (material_map_.insert( std::pair<std::string, int>(name, counter)).second == false) {
      throw std::runtime_error("the material " + name + " is specified twice in the configuration");
    }
    material_numbered_list_.push_back(name);
    printf("    %-6d %s\n", counter, name.c_str());
    counter++;
  }

//-----------------------------------------------------------------------------
// Read unit positions
//-----------------------------------------------------------------------------

  // TODO - use libconfig to check if this is a string or a group to allow
  // positions to be defined in the config file directly
  std::string position_filename = lattice_settings["positions"];
  std::ifstream position_file(position_filename.c_str());

  if(position_file.fail()) {
    throw std::runtime_error("failed to open position file " + position_filename);
  }

  printf("  unit cell positions (%s)\n", position_filename.c_str());

  counter = 0;

  std::string line;

  // read the motif into an array from the positions file
  while (getline(position_file, line)) {
    std::stringstream line_as_stream;
    std::pair<std::string, jblib::Vec3<double> > atom;

    line_as_stream.str(line);
    // read atom type name
    line_as_stream >> atom.first;

    // check the material type is defined
    if (material_map_.find(atom.first) == material_map_.end()) {
      throw std::runtime_error("material " + atom.first + " in the motif is not defined in the configuration");
    }

    // read atom coordinates
    line_as_stream >> atom.second.x >> atom.second.y >> atom.second.z;

    unit_cell_position_.push_back(atom);
    printf("    %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, atom.first.c_str(), atom.second.x, atom.second.y, atom.second.z);
    counter++;
  }
  position_file.close();

  calc_symmetry_operations();
}

void Lattice::init_lattice_positions(
  const libconfig::Setting &material_settings,
  const libconfig::Setting &lattice_settings)
{

  lattice_map_.resize(lattice_size_.x, lattice_size_.y, lattice_size_.z, unit_cell_position_.size());

  jblib::Vec3<int> kmesh_size(kpoints_.x*lattice_size_.x, kpoints_.y*lattice_size_.y, kpoints_.z*lattice_size_.z);
  if (!lattice_periodic_boundary_.x || !lattice_periodic_boundary_.y || !lattice_periodic_boundary_.z) {
    printf("\nzero padding non-periodic dimensions\n");
     // double any non-periodic dimensions for zero padding
    for (int i = 0; i < 3; ++i) {
      if (!lattice_periodic_boundary_[i]) {
        kmesh_size[i] = 2*kpoints_[i]*lattice_size_[i];
      }
    }
    printf("\npadded kspace size\n  %d  %d  %d\n", kmesh_size.x, kmesh_size.y, kmesh_size.z);
  }

  kspace_size_ = jblib::Vec3<int>(kmesh_size.x, kmesh_size.y, kmesh_size.z);
  kspace_map_.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z);

// initialize everything to -1 so we can check for double assignment below

  for (int i = 0, iend = product(lattice_size_)*unit_cell_position_.size(); i < iend; ++i) {
    lattice_map_[i] = -1;
  }

  for (int i = 0, iend = kspace_size_.x*kspace_size_.y*kspace_size_.z; i < iend; ++i) {
    kspace_map_[i] = -1;
  }

//-----------------------------------------------------------------------------
// Generate the realspace lattice positions
//-----------------------------------------------------------------------------

  int atom_counter = 0;
  rmax_.x = -FLT_MAX; rmax_.y = -FLT_MAX; rmax_.z = -FLT_MAX;
  rmin_.x = FLT_MAX; rmin_.y = FLT_MAX; rmin_.z = FLT_MAX;

  jblib::Vec3<int> translation_vector;
  lattice_super_cell_pos_.resize(num_unit_cell_positions()*product(lattice_size_));
  // loop over the translation vectors for lattice size
  for (int i = 0; i < lattice_size_.x; ++i) {
    for (int j = 0; j < lattice_size_.y; ++j) {
      for (int k = 0; k < lattice_size_.z; ++k) {
        // loop over atoms in the motif
        for (int m = 0, mend = unit_cell_position_.size(); m != mend; ++m) {
          translation_vector = jblib::Vec3<int>(i, j, k);

          // number the site in the fast integer lattice
          lattice_map_(i, j, k, m) = atom_counter;

          lattice_super_cell_pos_(atom_counter) = translation_vector;
          lattice_positions_.push_back(generate_position(unit_cell_position_[m].second, translation_vector));
          lattice_frac_positions_.push_back(generate_fractional_position(unit_cell_position_[m].second, translation_vector));
          lattice_materials_.push_back(unit_cell_position_[m].first);
          lattice_material_num_.push_back(material_map_[unit_cell_position_[m].first]);

          // store max coordinates
          for (int n = 0; n < 3; ++n) {
            if (lattice_positions_.back()[n] > rmax_[n]) {
              rmax_[n] = lattice_positions_.back()[n];
            }
            if (lattice_positions_.back()[n] < rmin_[n]) {
              rmin_[n] = lattice_positions_.back()[n];
            }
          }


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

  printf("\n----------------------------------------\n");
  printf("\ncomputed lattice positions (%d)\n", atom_counter);
  if (::output.is_verbose()) {
    for (int i = 0, iend = lattice_positions_.size(); i != iend; ++i) {
      printf("  %-6d %-6s % 3.6f % 3.6f % 3.6f %4d %4d %4d\n",
        i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z,
        lattice_super_cell_pos_(i).x, lattice_super_cell_pos_(i).y, lattice_super_cell_pos_(i).z);
    }
  } else {
    // avoid spamming the screen by default
    for (int i = 0; i < 8; ++i) {
    printf("  %-6d %-6s %3.6f % 3.6f % 3.6f\n",
      i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z);
  }
    if (lattice_positions_.size() > 0) {
      printf("  ... [use verbose output for details] ... \n");
    }
  }

//-----------------------------------------------------------------------------
// initialize global arrays
//-----------------------------------------------------------------------------
  globals::s.resize(globals::num_spins, 3);
  globals::ds_dt.resize(globals::num_spins, 3);

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

    printf("  reading initial spin configuration from: %s\n", spin_filename.c_str());

    load_spin_state_from_hdf5(spin_filename);
  }

  globals::h.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);
  // globals::wij.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z, 3, 3);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  // std::fill(globals::wij.data(), globals::wij.data()+kspace_size_.x*kspace_size_.y*kspace_size_.z*3*3, 0.0);

  material_count_.resize(num_materials(), 0);
  for (int i = 0; i < globals::num_spins; ++i) {
    int material_number = material_map_[lattice_materials_[i]];
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

  init_kspace();
}

void Lattice::init_kspace() {

  int i, j;
  printf("\n----------------------------------------\n");
  printf("\nreciprocal space\n");

  for (i = 0; i < 3; ++i) {
    kspace_size_[i] = lattice_size_[i];
  }

  printf("  kspace size\n    %4d %4d %4d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);

  kspace_map_.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z);

  double spg_lattice[3][3];
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = unit_cell_[i][j];
    }
  }

  double (*spg_positions)[3] = new double[unit_cell_position_.size()][3];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      spg_positions[i][j] = unit_cell_position_[i].second[j];
    }
  }

  int (*spg_types) = new int[unit_cell_position_.size()];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    spg_types[i] = material_map_[unit_cell_position_[i].first];
  }


  int num_mesh_points = kspace_size_.x*kspace_size_.y*kspace_size_.z;
  int (*grid_address)[3] = new int[num_mesh_points][3];
  int (*weights) = new int[num_mesh_points];

  int mesh[3] = {kspace_size_.x, kspace_size_.y, kspace_size_.z};
  int is_shift[] = {0, 0, 0};

  int num_ibz_points = spg_get_ir_reciprocal_mesh(grid_address,
                              kspace_map_.data(),
                              mesh,
                              is_shift,
                              1,
                              spg_lattice,
                              spg_positions,
                              spg_types,
                              unit_cell_position_.size(),
                              1e-5);

  printf("  irreducible kpoints\n    %d\n", num_ibz_points);

  jblib::Array<int,1> ibz_group;
  jblib::Array<int,1> ibz_index;
  jblib::Array<int,1> ibz_weight;

  // ibz_group maps from an ibz index to the grid index
  ibz_group.resize(num_ibz_points);

  // ibz_weight is the degeneracy of a point in the ibz
  ibz_weight.resize(num_ibz_points);

  // ibz_indez is the ibz_index from a grid index
  ibz_index.resize(num_mesh_points);

  // zero the weights array
  for (i = 0; i < num_mesh_points; ++i) {
      weights[i] = 0;
  }

  // calculate the weights
  for (i = 0; i < num_mesh_points; ++i) {
      weights[kspace_map_[i]]++;
  }

  // if weights[i] == 0 then it is not part of the irreducible group
  // so calculate the irreducible group of kpoints
  int counter = 0;
  for (i = 0; i < num_mesh_points; ++i) {
      if (weights[i] != 0) {
          ibz_group[counter] = i;
          ibz_weight[counter] = weights[i];
          ibz_index[i] = counter;
          counter++;
      } else {
          ibz_index[i] = ibz_index[kspace_map_[i]];
      }
  }

  if (is_debugging_enabled_) {
    std::ofstream ibz_file("debug_ibz.dat");
    for (int i = 0; i < num_mesh_points; ++i) {
      if (weights[i] != 0) {
        ibz_file << i << "\t" << grid_address[i][0] << "\t" << grid_address[i][1] << "\t" << grid_address[i][2] << std::endl;
      }
    }
    ibz_file.close();
  }

  if (is_debugging_enabled_) {
    std::ofstream kspace_file("kspace.dat");
    for (int i = 0; i < num_mesh_points; ++i) {
      // if (weights[i] != 0) {
        kspace_file << i << "\t" << grid_address[i][0] << "\t" << grid_address[i][1] << "\t" << grid_address[i][2] << std::endl;
      // }
    }
    kspace_file.close();
  }

  // find offset coordinates for unitcell

  double unitcell_offset[3] = {0.0, 0.0, 0.0};
  for (i = 0; i < unit_cell_position_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      if (unit_cell_position_[i].second[j] < unitcell_offset[j]){
        unitcell_offset[j] = unit_cell_position_[i].second[j];
      }
    }
  }

  printf("  unitcell offset (fractional)\n  % 6.6f % 6.6f % 6.6f",
    unitcell_offset[0], unitcell_offset[1], unitcell_offset[2]);

  kspace_inv_map_.resize(globals::num_spins, 3);

  for (i = 0; i < lattice_frac_positions_.size(); ++i) {
    jblib::Vec3<double> kvec;
    for (j = 0; j < 3; ++j) {
      kvec[j] = ((lattice_frac_positions_[i][j] - unitcell_offset[j])*kpoints_[j]);
    }
    // ::output.verbose("  kvec: % 3.6f % 3.6f % 3.6f\n", kvec.x, kvec.y, kvec.z);

    // check that the motif*kpoints is comsurate (within a tolerance) to the integer kspace_lattice
    //if (fabs(nint(kvec.x)-kvec.x) > 0.01 || fabs(nint(kvec.y)-kvec.y) > 0.01 || fabs(nint(kvec.z)-kvec.z) > 0.01) {
    //  jams_error("kpoint mesh does not map to the unit cell");
    //}
    // if (kspace_map_(nint(kvec.x), nint(kvec.y), nint(kvec.z)) != -1) {
    //   jams_error("attempted to assign multiple spins to the same point in the kspace map");
    // }
    for (j = 0; j < 3; ++j) {
      kspace_inv_map_(i, j) = nint(kvec[j]);
    }
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

void Lattice::calc_symmetry_operations() {
  printf("\n----------------------------------------\n");
  printf("\nsymmetry analysis\n");

  int i, j;
  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = unit_cell_[i][j];
    }
  }

  double (*spg_positions)[3] = new double[unit_cell_position_.size()][3];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      spg_positions[i][j] = unit_cell_position_[i].second[j];
    }
  }

  int (*spg_types) = new int[unit_cell_position_.size()];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    spg_types[i] = material_map_[unit_cell_position_[i].first];
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, unit_cell_position_.size(), 1e-5);

  printf("  International\n    %s (%d)\n", spglib_dataset_->international_symbol, spglib_dataset_->spacegroup_number );
  printf("  Hall symbol\n    %s\n", spglib_dataset_->hall_symbol );

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  printf("  point group\n    %s\n", ptsymbol);
  printf("  transformation matrix\n");
  for ( i = 0; i < 3; i++ ) {
      printf("    %f %f %f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }
  printf("  Wyckoff letters:\n");
  for ( i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      printf("    %c ", wl[spglib_dataset_->wyckoffs[i]]);
  }
  printf("\n");

  printf("  equivalent atoms:\n");
  for (i = 0; i < spglib_dataset_->n_atoms; i++) {
      printf("    %d ", spglib_dataset_->equivalent_atoms[i]);
  }
  printf("\n");

  output.verbose("  shifted lattice\n");
  output.verbose("    origin\n      % 3.6f % 3.6f % 3.6f\n",
    spglib_dataset_->origin_shift[0], spglib_dataset_->origin_shift[1], spglib_dataset_->origin_shift[2]);

  output.verbose("    lattice vectors\n");
  for (int i = 0; i < 3; ++i) {
    output.verbose("      % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }

  output.verbose("    positions\n");
  for (int i = 0; i < unit_cell_position_.size(); ++i) {
    double bij[3];
    matmul(spglib_dataset_->transformation_matrix, spg_positions[i], bij);
    output.verbose("  %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_numbered_list_[spg_types[i]].c_str(),
      bij[0], bij[1], bij[2]);
  }

  printf("  Bravais lattice\n");
  printf("    num atoms\n    %d\n", spglib_dataset_->n_std_atoms);
  printf("    lattice vectors\n");

  for (int i = 0; i < 3; ++i) {
    printf("  % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->std_lattice[i][0], spglib_dataset_->std_lattice[i][1], spglib_dataset_->std_lattice[i][2]);
  }

  int primitive_num_atoms = unit_cell_position_.size();
  double primitive_lattice[3][3];

  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      primitive_lattice[i][j] = spg_lattice[i][j];
    }
  }

  double (*primitive_positions)[3] = new double[unit_cell_position_.size()][3];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      primitive_positions[i][j] = spg_positions[i][j];
    }
  }

  int (*primitive_types) = new int[unit_cell_position_.size()];

  for (i = 0; i < unit_cell_position_.size(); ++i) {
    primitive_types[i] = spg_types[i];
  }

  primitive_num_atoms = spg_find_primitive(primitive_lattice, primitive_positions, primitive_types, unit_cell_position_.size(), 1e-5);

  // spg_find_primitive returns number of atoms in primitve cell
  if (primitive_num_atoms != unit_cell_position_.size()) {
    printf("\n");
    printf("unit cell is not a primitive cell\n");
    printf("\n");
    printf("  primitive lattice vectors:\n");

    for (int i = 0; i < 3; ++i) {
      printf("  % 3.6f % 3.6f % 3.6f\n",
        primitive_lattice[i][0], primitive_lattice[i][1], primitive_lattice[i][2]);
    }
    printf("\n");
    printf("  primitive motif positions:\n");

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      printf("  %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, material_numbered_list_[primitive_types[i]].c_str(),
        primitive_positions[i][0], primitive_positions[i][1], primitive_positions[i][2]);
      counter++;
    }
  }
}

// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(jblib::Vec3<int>& pos) const {
    for (int l = 0; l < 3; ++l) {
      if (!is_periodic(l) && (pos[l] < 0 || pos[l] >= lattice.num_unit_cells(l))) {
        return false;
      } else {
        pos[l] = (pos[l] + lattice.num_unit_cells(l))%lattice.num_unit_cells(l);
      }
    }
    return true;
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
