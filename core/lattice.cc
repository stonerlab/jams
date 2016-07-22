// Copyright 2014 Joseph Barker. All rights reserved.

extern "C"{
    #include "spglib/spglib.h"
}

#include "core/lattice.h"

#include <libconfig.h++>
#include <stdint.h>

#include <algorithm>
#include <stdexcept>
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
#include <functional>

#include "H5Cpp.h"

#include "core/consts.h"
#include "core/globals.h"
#include "core/exception.h"
#include "core/maths.h"
#include "core/sparsematrix.h"
#include "core/utils.h"
#include "core/neartree.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

using std::cout;
using std::endl;
using libconfig::Setting;
using libconfig::Config;

void Lattice::init_from_config(const libconfig::Config& cfg) {

  symops_enabled_ = true;
  cfg.lookupValue("lattice.symops", symops_enabled_);
  output.write("  symops: %s", symops_enabled_ ? "true" : "false");

  init_unit_cell(cfg.lookup("materials"), cfg.lookup("lattice"), cfg.lookup("unitcell"));

  if (symops_enabled_) {
    calc_symmetry_operations();
  }

  init_lattice_positions(cfg.lookup("materials"), cfg.lookup("lattice"));

  if (symops_enabled_) {
    init_kspace();
  }
}


void read_material_settings(Setting& cfg, Material &mat) {
  mat.name   = cfg["name"].c_str();
  mat.moment = double(cfg["moment"]);
  mat.gyro   = double(cfg["gyro"]);
  mat.alpha  = double(cfg["alpha"]);

  if (cfg.exists("transform")) {
    for (int i = 0; i < 3; ++i) {
      mat.transform[i] = double(cfg["transform"][i]);
    }
  } else {
    mat.transform = {1.0, 1.0, 1.0};
  }

  mat.randomize = false;

  if (cfg["spin"].getType() == libconfig::Setting::TypeString) {
    string spin_initializer = capitalize(cfg["spin"]);
    if (spin_initializer == "RANDOM") {
      mat.randomize = true;
    } else {
      jams_error("Unknown spin initializer %s selected", spin_initializer.c_str());
    }
  } else if (cfg["spin"].getType() == libconfig::Setting::TypeArray) {
    if (cfg["spin"].getLength() == 3) {
      for(int i = 0; i < 3; ++i) {
        mat.spin[i] = double(cfg["spin"][i]);
      }
    } else if (cfg["spin"].getLength() == 2) {
      // spin setting is spherical
      double theta = deg_to_rad(cfg["spin"][0]);
      double phi   = deg_to_rad(cfg["spin"][1]);
      mat.spin[0] = sin(theta)*cos(phi);
      mat.spin[1] = sin(theta)*sin(phi);
      mat.spin[2] = cos(theta);
    } else {
      throw general_exception("material spin array is not of a recognised size (2 or 3)", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
  }

}

Lattice::Lattice() {

}

Lattice::~Lattice() {
  if (spglib_dataset_ != NULL) {
    spg_free_dataset(spglib_dataset_);
  }
}

Vec3 Lattice::generate_position(
  const Vec3 unit_cell_frac_pos,
  const Vec3i translation_vector) const
{
  return super_cell.unit_cell * generate_fractional_position(unit_cell_frac_pos, translation_vector);
}

// generate a position within a periodic image of the entire system
Vec3 Lattice::generate_image_position(
  const Vec3 unit_cell_cart_pos,
  const Vec3i image_vector) const
{
  Vec3 frac_pos = cartesian_to_fractional(unit_cell_cart_pos);
  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      frac_pos[n] = frac_pos[n] + image_vector[n] * super_cell.size[n];
    }
  }
  return fractional_to_cartesian(frac_pos);
}

Vec3 Lattice::generate_fractional_position(
  const Vec3 unit_cell_frac_pos,
  const Vec3i translation_vector) const
{
  return Vec3(unit_cell_frac_pos.x + translation_vector.x,
                             unit_cell_frac_pos.y + translation_vector.y,
                             unit_cell_frac_pos.z + translation_vector.z);
}

void Lattice::read_motif_from_config(const libconfig::Setting &positions) {
  Atom atom;
  string atom_name;

  motif_.clear();

  for (int i = 0; i < positions.getLength(); ++i) {

    atom_name = positions[i][0].c_str();

    // check the material type is defined
    if (material_name_map_.find(atom_name) == material_name_map_.end()) {
      throw general_exception("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material = material_name_map_[atom_name].id;

    atom.pos.x = positions[i][1][0];
    atom.pos.y = positions[i][1][1];
    atom.pos.z = positions[i][1][2];

    atom.id = motif_.size();

    motif_.push_back(atom);
  }
}

void Lattice::read_motif_from_file(const std::string &filename) {
  std::string line;
  std::ifstream position_file(filename.c_str());

  if(position_file.fail()) {
    throw general_exception("failed to open position file " + filename, __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  motif_.clear();

  // read the motif into an array from the positions file
  while (getline(position_file, line)) {
    if(string_is_comment(line)) {
      continue;
    }
    std::stringstream line_as_stream;
    string atom_name;
    Atom atom;

    line_as_stream.str(line);

    // read atom type name
    line_as_stream >> atom_name >> atom.pos.x >> atom.pos.y >> atom.pos.z;

    // check the material type is defined
    if (material_name_map_.find(atom_name) == material_name_map_.end()) {
      throw general_exception("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material = material_name_map_[atom_name].id;
    atom.id = motif_.size();

    motif_.push_back(atom);
  }
  position_file.close();
}


void Lattice::init_unit_cell(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings) {
  using namespace globals;
  using std::string;
  using std::pair;

  int i, j;


//-----------------------------------------------------------------------------
// Read lattice vectors
//-----------------------------------------------------------------------------

  // We transpose during the read because the unit cell matrix must have the
  // lattice vectors as the columns but it is easiest to define each vector in
  // the input
  // | a1x a2x a2x |  | A |   | A.a1x + B.a2x + C.a3x |
  // | a1y a2y a3y |  | B | = | A.a1y + B.a2y + C.a3y |
  // | a1z a2z a3z |  | C |   | A.a1z + B.a2z + C.a3z |
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      super_cell.unit_cell[i][j] = unitcell_settings["basis"][i][j];
    }
  }
  output.write("\n----------------------------------------\n");
  output.write("\nunit cell\n");

  output.write("  lattice vectors\n");
  for (i = 0; i < 3; ++i) {
    output.write("    % 3.6f % 3.6f % 3.6f\n",
      super_cell.unit_cell[i][0], super_cell.unit_cell[i][1], super_cell.unit_cell[i][2]);
  }

  super_cell.unit_cell_inv = super_cell.unit_cell.inverse();

  output.write("  inverse lattice vectors\n");
  for (i = 0; i < 3; ++i) {
    output.write("    % 3.6f % 3.6f % 3.6f\n",
      super_cell.unit_cell_inv[i][0], super_cell.unit_cell_inv[i][1], super_cell.unit_cell_inv[i][2]);
  }

  super_cell.parameter = unitcell_settings["parameter"];
  output.write("  lattice parameter (m):\n    %3.6e\n", super_cell.parameter);

  if (super_cell.parameter < 0.0) {
    throw general_exception("lattice parameter cannot be negative", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  output.write("  unitcell volume (m^3):\n    %3.6e\n", this->volume());

  for (i = 0; i < 3; ++i) {
    super_cell.size[i] = lattice_settings["size"][i];
  }

  output.write("  lattice size\n    %d  %d  %d\n",
    super_cell.size.x, super_cell.size.y, super_cell.size.z);

//-----------------------------------------------------------------------------
// Read boundary conditions
//-----------------------------------------------------------------------------

  if(!lattice_settings.exists("periodic")) {
    // sane default
    output.write(
      "\nASSUMPTION: no boundary conditions specified - assuming 3D periodic\n");
    for (i = 0; i < 3; ++i) {
      super_cell.periodic[i] = true;
    }
  } else {
    for (i = 0; i < 3; ++i) {
      super_cell.periodic[i] = lattice_settings["periodic"][i];
    }
  }
  output.write("  boundary conditions\n    %s  %s  %s\n",
    super_cell.periodic.x ? "periodic" : "open",
    super_cell.periodic.y ? "periodic" : "open",
    super_cell.periodic.z ? "periodic" : "open");

  metric_ = new DistanceMetric(super_cell.unit_cell, super_cell.size, super_cell.periodic);

//-----------------------------------------------------------------------------
// Read materials
//-----------------------------------------------------------------------------

  output.write("  materials\n");

  Material material;
  for (i = 0; i < material_settings.getLength(); ++i) {
    material.id = i;
    read_material_settings(material_settings[i], material);
    if (material_name_map_.insert({material.name, material}).second == false) {
      throw std::runtime_error("the material " + material.name + " is specified twice in the configuration");
    }
    material_id_map_.insert({material.id, material});
    output.write("    %-6d %s\n", material.id, material.name.c_str());
  }

//-----------------------------------------------------------------------------
// Read unit positions
//-----------------------------------------------------------------------------

  // TODO - use libconfig to check if this is a string or a group to allow
  // positions to be defined in the config file directly

  std::string position_filename;
  if (unitcell_settings["positions"].isList()) {
    position_filename = seedname + ".cfg";
    read_motif_from_config(unitcell_settings["positions"]);
  } else {
     position_filename = unitcell_settings["positions"].c_str();
    read_motif_from_file(position_filename);
  }

  output.write("  unit cell positions (%s)\n", position_filename.c_str());

  for (const Atom &atom: motif_) {
    output.write("    %-6d %s % 3.6f % 3.6f % 3.6f\n", atom.id, material_name(atom.material).c_str(), atom.pos.x, atom.pos.y, atom.pos.z);
  }
}

void Lattice::init_lattice_positions(
  const libconfig::Setting &material_settings,
  const libconfig::Setting &lattice_settings)
{

  lattice_map_.resize(super_cell.size.x, super_cell.size.y, super_cell.size.z, motif_.size());

  Vec3i kmesh_size(kpoints_.x*super_cell.size.x, kpoints_.y*super_cell.size.y, kpoints_.z*super_cell.size.z);
  if (!super_cell.periodic.x || !super_cell.periodic.y || !super_cell.periodic.z) {
    output.write("\nzero padding non-periodic dimensions\n");
     // double any non-periodic dimensions for zero padding
    for (int i = 0; i < 3; ++i) {
      if (!super_cell.periodic[i]) {
        kmesh_size[i] = 2*kpoints_[i]*super_cell.size[i];
      }
    }
    output.write("\npadded kspace size\n  %d  %d  %d\n", kmesh_size.x, kmesh_size.y, kmesh_size.z);
  }

  kspace_size_ = Vec3i(kmesh_size.x, kmesh_size.y, kmesh_size.z);
  kspace_map_.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z);

// initialize everything to -1 so we can check for double assignment below

  for (int i = 0, iend = product(super_cell.size)*motif_.size(); i < iend; ++i) {
    lattice_map_[i] = -1;
  }

  for (int i = 0, iend = kspace_size_.x*kspace_size_.y*kspace_size_.z; i < iend; ++i) {
    kspace_map_[i] = -1;
  }

//-----------------------------------------------------------------------------
// Generate the realspace lattice positions
//-----------------------------------------------------------------------------

  int atom_counter = 0;
  rmax_.x = -DBL_MAX; rmax_.y = -DBL_MAX; rmax_.z = -DBL_MAX;
  rmin_.x = DBL_MAX; rmin_.y = DBL_MAX; rmin_.z = DBL_MAX;

  Vec3i translation_vector;
  lattice_super_cell_pos_.resize(num_unit_cell_positions() * product(super_cell.size));

  // loop over the translation vectors for lattice size
  for (int i = 0; i < super_cell.size.x; ++i) {
    for (int j = 0; j < super_cell.size.y; ++j) {
      for (int k = 0; k < super_cell.size.z; ++k) {

        translation_vector = {i, j, k};

        // loop over atoms in the motif
        for (int m = 0, mend = motif_.size(); m != mend; ++m) {

          // number the site in the fast integer lattice
          lattice_map_(i, j, k, m) = atom_counter;

          lattice_super_cell_pos_(atom_counter) = translation_vector;
          lattice_positions_.push_back(generate_position(motif_[m].pos, translation_vector));
          lattice_frac_positions_.push_back(generate_fractional_position(motif_[m].pos, translation_vector));
          lattice_materials_.push_back(material_id_map_[motif_[m].material].name);
          lattice_material_num_.push_back(motif_[m].material);

          atoms_.push_back({atom_counter, motif_[m].material, generate_position(motif_[m].pos, translation_vector)});

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

  // populate the NearTree
  neartree_ = new NearTree<Atom, DistanceMetric>(*metric_, atoms_);

  globals::num_spins = atom_counter;
  globals::num_spins3 = 3*atom_counter;

  output.write("\n----------------------------------------\n");
  output.write("\ncomputed lattice positions (%d)\n", atom_counter);
  if (::output.is_verbose()) {
    for (int i = 0, iend = lattice_positions_.size(); i != iend; ++i) {
      output.write("  %-6d %-6s % 3.6f % 3.6f % 3.6f %4d %4d %4d\n",
        i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z,
        lattice_super_cell_pos_(i).x, lattice_super_cell_pos_(i).y, lattice_super_cell_pos_(i).z);
    }
  } else {
    // avoid spamming the screen by default
    for (int i = 0; i < 8; ++i) {
    output.write("  %-6d %-6s %3.6f % 3.6f % 3.6f\n",
      i, lattice_materials_[i].c_str(), lattice_positions_[i].x, lattice_positions_[i].y, lattice_positions_[i].z);
  }
    if (lattice_positions_.size() > 0) {
      output.write("  ... [use verbose output for details] ... \n");
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

    output.write("  reading initial spin configuration from: %s\n", spin_filename.c_str());

    load_spin_state_from_hdf5(spin_filename);
  }

  globals::h.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);
  // globals::wij.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z, 3, 3);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  // std::fill(globals::wij.data(), globals::wij.data()+kspace_size_.x*kspace_size_.y*kspace_size_.z*3*3, 0.0);

  num_of_material_.resize(num_materials(), 0);
  for (int i = 0; i < globals::num_spins; ++i) {
    int material_number = material_name_map_[lattice_materials_[i]].id;
    num_of_material_[material_number]++;

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

void Lattice::init_kspace() {

  if (!symops_enabled_) {
    throw general_exception("Lattice::init_kspace() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  int i, j;
  output.write("\n----------------------------------------\n");
  output.write("\nreciprocal space\n");

  for (i = 0; i < 3; ++i) {
    kspace_size_[i] = super_cell.size[i];
  }

  output.write("  kspace size\n    %4d %4d %4d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);

  kspace_map_.resize(kspace_size_.x, kspace_size_.y, kspace_size_.z);

  double spg_lattice[3][3];
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = super_cell.unit_cell[i][j];
    }
  }

  double (*spg_positions)[3] = new double[motif_.size()][3];

  for (i = 0; i < motif_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      spg_positions[i][j] = motif_[i].pos[j];
    }
  }

  int (*spg_types) = new int[motif_.size()];

  for (i = 0; i < motif_.size(); ++i) {
    spg_types[i] = motif_[i].material;
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
                              motif_.size(),
                              1e-5);

  output.write("  irreducible kpoints\n    %d\n", num_ibz_points);

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
  for (i = 0; i < motif_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      if (motif_[i].pos[j] < unitcell_offset[j]){
        unitcell_offset[j] = motif_[i].pos[j];
      }
    }
  }

  output.write("  unitcell offset (fractional)\n  % 6.6f % 6.6f % 6.6f",
    unitcell_offset[0], unitcell_offset[1], unitcell_offset[2]);

  kspace_inv_map_.resize(globals::num_spins, 3);

  for (i = 0; i < lattice_frac_positions_.size(); ++i) {
    Vec3 kvec;
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

  if (!symops_enabled_) {
    throw general_exception("Lattice::calc_symmetry_operations() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  output.write("\n----------------------------------------\n");
  output.write("\nsymmetry analysis\n");

  int i, j;
  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = super_cell.unit_cell[i][j];
    }
  }

  double (*spg_positions)[3] = new double[motif_.size()][3];

  for (i = 0; i < motif_.size(); ++i) {
    for (j = 0; j < 3; ++j) {
      spg_positions[i][j] = motif_[i].pos[j];
    }
  }

  int (*spg_types) = new int[motif_.size()];

  for (i = 0; i < motif_.size(); ++i) {
    spg_types[i] = motif_[i].material;
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, motif_.size(), 1e-5);

  output.write("  International\n    %s (%d)\n", spglib_dataset_->international_symbol, spglib_dataset_->spacegroup_number );
  output.write("  Hall symbol\n    %s\n", spglib_dataset_->hall_symbol );
  output.write("  Hall number\n    %d\n", spglib_dataset_->hall_number );

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  output.write("  point group\n    %s\n", ptsymbol);
  output.write("  transformation matrix\n");
  for ( i = 0; i < 3; i++ ) {
      output.write("    %f %f %f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }
  output.write("  Wyckoff letters:\n");
  for ( i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      output.write("    %c ", wl[spglib_dataset_->wyckoffs[i]]);
  }
  output.write("\n");

  output.write("  equivalent atoms:\n");
  for (i = 0; i < spglib_dataset_->n_atoms; i++) {
      output.write("    %d ", spglib_dataset_->equivalent_atoms[i]);
  }
  output.write("\n");

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
  for (int i = 0; i < motif_.size(); ++i) {
    double bij[3];
    matmul(spglib_dataset_->transformation_matrix, spg_positions[i], bij);
    output.verbose("  %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_id_map_[spg_types[i]].name.c_str(),
      bij[0], bij[1], bij[2]);
  }

  output.write("  Standard lattice\n");
  output.write("    std lattice vectors\n");

  for (int i = 0; i < 3; ++i) {
    output.write("  % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->std_lattice[i][0], spglib_dataset_->std_lattice[i][1], spglib_dataset_->std_lattice[i][2]);
  }
  output.write("    num std atoms\n    %d\n", spglib_dataset_->n_std_atoms);

  output.write("    std_positions\n");
  for (int i = 0; i < spglib_dataset_->n_std_atoms; ++i) {
    output.write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_id_map_[spglib_dataset_->std_types[i]].name.c_str(),
      spglib_dataset_->std_positions[i][0], spglib_dataset_->std_positions[i][1], spglib_dataset_->std_positions[i][2]);
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

  // spg_find_primitive returns number of atoms in primitve cell
  if (primitive_num_atoms != motif_.size()) {
    output.write("\n");
    output.write("unit cell is not a primitive cell\n");
    output.write("\n");
    output.write("  primitive lattice vectors:\n");

    for (int i = 0; i < 3; ++i) {
      output.write("  % 3.6f % 3.6f % 3.6f\n",
        primitive_lattice[i][0], primitive_lattice[i][1], primitive_lattice[i][2]);
    }
    output.write("\n");
    output.write("  primitive motif positions:\n");

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      output.write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, material_id_map_[primitive_types[i]].name.c_str(),
        primitive_positions[i][0], primitive_positions[i][1], primitive_positions[i][2]);
      counter++;
    }
  }

  output.write("\n");
  output.write("  Symmetry operations\n");
  output.write("    num symops\n    %d\n", spglib_dataset_->n_operations);

  Mat3 rot;
  Mat3 id(1, 0, 0, 0, 1, 0, 0, 0, 1);

  for (int i = 0; i < spglib_dataset_->n_operations; ++i) {

    output.verbose("%d\n---\n", i);
    output.verbose("%8d  %8d  %8d\n%8d  %8d  %8d\n%8d  %8d  %8d\n",
      spglib_dataset_->rotations[i][0][0], spglib_dataset_->rotations[i][0][1], spglib_dataset_->rotations[i][0][2],
      spglib_dataset_->rotations[i][1][0], spglib_dataset_->rotations[i][1][1], spglib_dataset_->rotations[i][1][2],
      spglib_dataset_->rotations[i][2][0], spglib_dataset_->rotations[i][2][1], spglib_dataset_->rotations[i][2][2]);
    // std::cout << i << "\t" << spglib_dataset_->translations[i][0] << "\t" << spglib_dataset_->translations[i][1] << "\t" << spglib_dataset_->translations[i][2] << std::endl;

    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        rot[m][n] = spglib_dataset_->rotations[i][m][n];
      }
    }

    rotations_.push_back(rot);
  }
}


void Lattice::set_spacegroup(const int hall_number) {
  Mat3 rot;

  int spg_n_operations = 0;
  int spg_rotations[192][3][3];
  double spg_translations[192][3];
  spg_n_operations = spg_get_symmetry_from_database(spg_rotations, spg_translations, hall_number);

  for (int i = 0; i < spg_n_operations; ++i) {
    std::cout << i << "\t" << spg_translations[i][0] << "\t" << spg_translations[i][1] << "\t" << spg_translations[i][2] << std::endl;

    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        rot[m][n] = spg_rotations[i][m][n];
      }
    }

    rotations_.push_back(rot);
  }
}


// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(Vec3i& pos) const {
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
  Vec3i pos3(pos.x, pos.y, pos.z);
  bool is_within_lattice = apply_boundary_conditions(pos3);
  if (is_within_lattice) {
    pos.x = pos3.x;
    pos.y = pos3.y;
    pos.z = pos3.z;
  }
  return is_within_lattice;
}


// void Lattice::atom_nearest_neighbours(const int i, const double r_cutoff, std::vector<Atom> &neighbours) {
//   const double eps = kEps;

//   neartree_.find_in_radius(r_cutoff, neighbours, {i, lattice.atom_material(i), lattice.atom_position(i)});


//   const double r_min = (*std::min_element(neighbours[i].begin(), neighbours[i].end()));

//     auto nbr_end = std::remove_if(neighbour_list[i].begin(), neighbour_list[i].end(), [r_min, eps](const Neighbour& nbr) {
//       return std::abs(nbr.distance - r_min) > eps;
//     });

//     neighbour_list[i].erase(nbr_end, neighbour_list[i].end());
// }
