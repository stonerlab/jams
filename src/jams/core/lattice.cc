// Copyright 2014 Joseph Barker. All rights reserved.

extern "C"{
    #include "spglib/spglib.h"
}

#include "jams/core/lattice.h"

#include <libconfig.h++>
#include <cstddef>

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <cfloat>
#include <jams/core/defaults.h>

#include "H5Cpp.h"

#include "jams/core/error.h"
#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/globals.h"
#include "jams/core/exception.h"
#include "jams/core/maths.h"
#include "jams/core/utils.h"
#include "jams/core/neartree.h"
#include "jblib/containers/vec.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

using std::cout;
using std::endl;
using libconfig::Setting;
using libconfig::Config;

namespace {
    Vec3 shift_fractional_coordinate_to_zero_one(Vec3 r) {
      for (auto n = 0; n < 3; ++n) {
        if (r[n] < 0.0) {
          r[n] = r[n] + 1.0;
        }
      }
      return r;
    }

    bool is_fractional_coordinate_valid(const Vec3 &r) {
      for (auto n = 0; n < 3; ++n) {
        if (r[n] < 0.0 || r[n] > 1.0) {
          return false;
        }
      }
      return true;
    }

    double rhombus_inradius(const Vec3& v1, const Vec3& v2) {
      return abs(cross(v1, v2)) / (2.0 * abs(v2));
    }

    double rhombohedron_inradius(const Vec3& v1, const Vec3& v2, const Vec3& v3) {
      return std::min(rhombus_inradius(v1, v2), std::min(rhombus_inradius(v3, v1), rhombus_inradius(v2, v3)));
    }
}

namespace jams {
    Mat3 unit_cell_matrix(const Vec3& a1, const Vec3& a2,  const Vec3& a3) {
      return {a1[0], a2[0], a3[0], a1[1], a2[1], a3[1], a1[2], a2[2], a3[2]};
    }
    Mat3 inverse_unit_cell_matrix(const Mat3& unit_cell_matrix) {
      return inverse(unit_cell_matrix);
    }
}

void Lattice::init_from_config(const libconfig::Config& cfg) {

  symops_enabled_ = true;
  cfg.lookupValue("lattice.symops", symops_enabled_);
  output->write("  symops: %s", symops_enabled_ ? "true" : "false");

  init_unit_cell(cfg.lookup("materials"), cfg.lookup("lattice"), cfg.lookup("unitcell"));

  if (symops_enabled_) {
    calc_symmetry_operations();
  }

  init_lattice_positions(cfg.lookup("materials"), cfg.lookup("lattice"));
}


void read_material_settings(Setting& cfg, Material &mat) {
  mat.name      = cfg["name"].c_str();
  mat.moment    = double(cfg["moment"]);
  mat.gyro      = jams::default_gyro;
  mat.alpha     = jams::default_alpha;
  mat.spin      = jams::default_spin;
  mat.transform = jams::default_spin_transform;
  mat.randomize = false;

  cfg.lookupValue("gyro", mat.gyro);
  cfg.lookupValue("alpha", mat.alpha);

  if (cfg.exists("transform")) {
    for (auto i = 0; i < 3; ++i) {
      mat.transform[i] = double(cfg["transform"][i]);
    }
  }


  if (cfg.exists("spin")) {
    if (cfg["spin"].getType() == libconfig::Setting::TypeString) {
      string spin_initializer = capitalize(cfg["spin"]);
      if (spin_initializer == "RANDOM") {
        mat.randomize = true;
      } else {
        jams_error("Unknown spin initializer %s selected", spin_initializer.c_str());
      }
    } else if (cfg["spin"].getType() == libconfig::Setting::TypeArray) {
      if (cfg["spin"].getLength() == 3) {
        for (int i = 0; i < 3; ++i) {
          mat.spin[i] = double(cfg["spin"][i]);
        }
      } else if (cfg["spin"].getLength() == 2) {
        // spin setting is spherical
        double theta = deg_to_rad(cfg["spin"][0]);
        double phi = deg_to_rad(cfg["spin"][1]);
        mat.spin[0] = sin(theta) * cos(phi);
        mat.spin[1] = sin(theta) * sin(phi);
        mat.spin[2] = cos(theta);
      } else {
        throw general_exception("material spin array is not of a recognised size (2 or 3)", __FILE__, __LINE__,
                                __PRETTY_FUNCTION__);
      }
    }
  }

  if (!equal(abs(mat.spin), 1.0)) {
    throw general_exception("material spin is not of unit length", __FILE__, __LINE__, __PRETTY_FUNCTION__);
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
  return {unit_cell_frac_pos[0] + translation_vector[0],
                             unit_cell_frac_pos[1] + translation_vector[1],
                             unit_cell_frac_pos[2] + translation_vector[2]};
}

void Lattice::read_motif_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format) {
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

    atom.pos[0] = positions[i][1][0];
    atom.pos[1] = positions[i][1][1];
    atom.pos[2] = positions[i][1][2];

    if (coordinate_format == CoordinateFormat::Cartesian) {
      atom.pos = cartesian_to_fractional(atom.pos);
    }

    atom.pos = shift_fractional_coordinate_to_zero_one(atom.pos);

    if (!is_fractional_coordinate_valid(atom.pos)) {
      throw std::runtime_error("atom position " + std::to_string(i) + " is not a valid fractional coordinate");
    }

    atom.id = motif_.size();

    motif_.push_back(atom);
  }
}

void Lattice::read_motif_from_file(const std::string &filename, CoordinateFormat coordinate_format) {
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
    line_as_stream >> atom_name >> atom.pos[0] >> atom.pos[1] >> atom.pos[2];

    if (coordinate_format == CoordinateFormat::Cartesian) {
      atom.pos = cartesian_to_fractional(atom.pos);
    }

    atom.pos = shift_fractional_coordinate_to_zero_one(atom.pos);

    if (!is_fractional_coordinate_valid(atom.pos)) {
      throw std::runtime_error("atom position " + std::to_string(motif_.size()) + " is not a valid fractional coordinate");
    }
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

  // unit cell matrix is made of a,b,c lattice vectors as
  //
  // a_x  b_x  c_x
  // a_y  b_y  c_y
  // a_z  b_z  c_z
  //
  // this is consistent with the definition used by spglib


  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      super_cell.unit_cell[i][j] = unitcell_settings["basis"][i][j];
    }
  }
  output->write("\n----------------------------------------\n");
  output->write("\nunit cell\n");

  output->write("  lattice vectors\n");
  output->write("    a = (%f, %f, %f)\n", super_cell.unit_cell[0][0], super_cell.unit_cell[1][0], super_cell.unit_cell[2][0]);
  output->write("    b = (%f, %f, %f)\n", super_cell.unit_cell[0][1], super_cell.unit_cell[1][1], super_cell.unit_cell[2][1]);
  output->write("    c = (%f, %f, %f)\n", super_cell.unit_cell[0][2], super_cell.unit_cell[1][2], super_cell.unit_cell[2][2]);
  output->write("\n");

  output->write("  lattice vectors (matrix form)\n");

  for (i = 0; i < 3; ++i) {
    output->write("    % 3.6f % 3.6f % 3.6f\n",
      super_cell.unit_cell[i][0], super_cell.unit_cell[i][1], super_cell.unit_cell[i][2]);
  }

  super_cell.unit_cell_inv = jams::inverse_unit_cell_matrix(super_cell.unit_cell);

  output->write("  inverse lattice vectors (matrix form)\n");
  for (i = 0; i < 3; ++i) {
    output->write("    % 3.6f % 3.6f % 3.6f\n",
      super_cell.unit_cell_inv[i][0], super_cell.unit_cell_inv[i][1], super_cell.unit_cell_inv[i][2]);
  }

  super_cell.parameter = unitcell_settings["parameter"];
  output->write("  lattice parameter (m):\n    %3.6e\n", super_cell.parameter);

  if (super_cell.parameter < 0.0) {
    throw general_exception("lattice parameter cannot be negative", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if (super_cell.parameter > 1e-7) {
    jams_warning("lattice parameter is unusually large - units should be meters");
  }

  output->write("  unitcell volume (m^3):\n    %3.6e\n", this->volume());

  const double original_volume = std::abs(determinant(super_cell.unit_cell));

  for (i = 0; i < 3; ++i) {
    super_cell.size[i] = lattice_settings["size"][i];
  }

  output->write("  lattice size\n    %d  %d  %d\n",
                super_cell.size[0], super_cell.size[1], super_cell.size[2]);

  if (lattice_settings.exists("global_rotation") && lattice_settings.exists("orientation_axis")) {
    jams_warning("Orientation and global rotation are both in config. Orientation will be applied first and then global rotation.");
  }


  if (lattice_settings.exists("orientation_axis")) {
    Vec3 orientation_axis = {0, 0, 1};
    Vec3 orientation_lattice_vector = {0, 0, 1};

    for (auto n = 0; n < 3; ++n) {
      orientation_axis[n] = lattice_settings["orientation_axis"][n];
    }

    for (auto n = 0; n < 3; ++n) {
      orientation_lattice_vector[n] = lattice_settings["orientation_lattice_vector"][n];
    }

    Vec3 orientation_cartesian_vector = normalize(super_cell.unit_cell * orientation_lattice_vector);

    output->write("  orientation_axis\n");
    output->write("    % 3.6f % 3.6f % 3.6f\n", orientation_axis[0], orientation_axis[1], orientation_axis[2]);

    output->write("  orientation_lattice_vector\n");
    output->write("    % 3.6f % 3.6f % 3.6f\n", orientation_lattice_vector[0], orientation_lattice_vector[1], orientation_lattice_vector[2]);

    output->write("  orientation_cartesian_vector\n");
    output->write("    % 3.6f % 3.6f % 3.6f\n", orientation_cartesian_vector[0], orientation_cartesian_vector[1], orientation_cartesian_vector[2]);

    Mat3 orientation_matrix = rotation_matrix_between_vectors(orientation_cartesian_vector, orientation_axis);

    output->write("  orientation rotation matrix\n");
    output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[0][0], orientation_matrix[0][1], orientation_matrix[0][2]);
    output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[1][0], orientation_matrix[1][1], orientation_matrix[1][2]);
    output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[2][0], orientation_matrix[2][1], orientation_matrix[2][2]);
    output->write("\n");

    Vec3 rotated_orientation_vector = orientation_matrix * orientation_cartesian_vector;

    output->verbose("  rotated_orientation_vector\n");
    output->verbose("    % 3.6f % 3.6f % 3.6f\n", rotated_orientation_vector[0], rotated_orientation_vector[1], rotated_orientation_vector[2]);

    Vec3 orient_a = orientation_matrix * Vec3{super_cell.unit_cell[0][0], super_cell.unit_cell[1][0], super_cell.unit_cell[2][0]};
    Vec3 orient_b = orientation_matrix * Vec3{super_cell.unit_cell[0][1], super_cell.unit_cell[1][1], super_cell.unit_cell[2][1]};
    Vec3 orient_c = orientation_matrix * Vec3{super_cell.unit_cell[0][2], super_cell.unit_cell[1][2], super_cell.unit_cell[2][2]};

    output->write("  oriented lattice vectors\n");
    output->write("    a = (%f, %f, %f)\n", orient_a[0], orient_a[1], orient_a[2]);
    output->write("    b = (%f, %f, %f)\n", orient_b[0], orient_b[1], orient_b[2]);
    output->write("    c = (%f, %f, %f)\n", orient_c[0], orient_c[1], orient_c[2]);
    output->write("\n");

    super_cell.unit_cell = matrix_from_cols(orient_a, orient_b, orient_c);
    super_cell.unit_cell_inv = jams::inverse_unit_cell_matrix(super_cell.unit_cell);

    output->write("  unitcell volume (m^3):\n    %3.6e\n", this->volume());

    const double rotated_volume = std::abs(determinant(super_cell.unit_cell));
    if (std::abs(rotated_volume - original_volume) > 1e-8) {
      jams_error("Volume has changed after rotation of the unit cell");
    }
  }



  if (lattice_settings.exists("global_rotation")) {
    Mat3 global_rotation_matrix = kIdentityMat3;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        global_rotation_matrix[i][j] = lattice_settings["global_rotation"][i][j];
      }
    }

    Vec3 orient_a = global_rotation_matrix * Vec3{super_cell.unit_cell[0][0], super_cell.unit_cell[1][0], super_cell.unit_cell[2][0]};
    Vec3 orient_b = global_rotation_matrix * Vec3{super_cell.unit_cell[0][1], super_cell.unit_cell[1][1], super_cell.unit_cell[2][1]};
    Vec3 orient_c = global_rotation_matrix * Vec3{super_cell.unit_cell[0][2], super_cell.unit_cell[1][2], super_cell.unit_cell[2][2]};

    output->write("  global rotated lattice vectors\n");
    output->write("    a = (%f, %f, %f)\n", orient_a[0], orient_a[1], orient_a[2]);
    output->write("    b = (%f, %f, %f)\n", orient_b[0], orient_b[1], orient_b[2]);
    output->write("    c = (%f, %f, %f)\n", orient_c[0], orient_c[1], orient_c[2]);
    output->write("\n");

    super_cell.unit_cell = jams::unit_cell_matrix(orient_a, orient_b, orient_c);
    super_cell.unit_cell_inv = jams::inverse_unit_cell_matrix(super_cell.unit_cell);

    output->write("  unitcell volume (m^3):\n    %3.6e\n", this->volume());

    const double rotated_volume = std::abs(determinant(super_cell.unit_cell));
    if (std::abs(rotated_volume - original_volume) > 1e-8) {
      jams_error("Volume has changed after rotation of the unit cell");
    }
  }

//-----------------------------------------------------------------------------
// Read boundary conditions
//-----------------------------------------------------------------------------

  super_cell.periodic = jams::default_lattice_periodic_boundaries;
  if(lattice_settings.exists("periodic")) {
    for (i = 0; i < 3; ++i) {
      super_cell.periodic[i] = lattice_settings["periodic"][i];
    }
  }
  output->write("  boundary conditions\n    %s  %s  %s\n",
    super_cell.periodic[0] ? "periodic" : "open",
    super_cell.periodic[1] ? "periodic" : "open",
    super_cell.periodic[2] ? "periodic" : "open");

  metric_ = new DistanceMetric(super_cell.unit_cell, super_cell.size, super_cell.periodic);

//-----------------------------------------------------------------------------
// Read materials
//-----------------------------------------------------------------------------

  output->write("  materials\n");

  Material material;
  for (i = 0; i < material_settings.getLength(); ++i) {
    material.id = i;
    read_material_settings(material_settings[i], material);
    if (material_name_map_.insert({material.name, material}).second == false) {
      throw std::runtime_error("the material " + material.name + " is specified twice in the configuration");
    }
    material_id_map_.insert({material.id, material});
    output->write("    %-6d %s\n", material.id, material.name.c_str());
  }

//-----------------------------------------------------------------------------
// Read unit positions
//-----------------------------------------------------------------------------

  // TODO - use libconfig to check if this is a string or a group to allow
  // positions to be defined in the config file directly

  CoordinateFormat cfg_coordinate_format = CoordinateFormat::Fractional;

  std::string cfg_coordinate_format_name = "FRACTIONAL";
  unitcell_settings.lookupValue("format", cfg_coordinate_format_name);

  if (capitalize(cfg_coordinate_format_name) == "FRACTIONAL") {
    cfg_coordinate_format = CoordinateFormat::Fractional;
  } else if (capitalize(cfg_coordinate_format_name) == "CARTESIAN") {
    cfg_coordinate_format = CoordinateFormat::Cartesian;
  } else {
    throw std::runtime_error("Unknown coordinate format for atom positions in unit cell");
  }

  std::string position_filename;
  if (unitcell_settings["positions"].isList()) {
    position_filename = seedname + ".cfg";
    read_motif_from_config(unitcell_settings["positions"], cfg_coordinate_format);
  } else {
     position_filename = unitcell_settings["positions"].c_str();
    read_motif_from_file(position_filename, cfg_coordinate_format);
  }

  output->write("  unit cell positions (%s)\n", position_filename.c_str());

  output->write("  format: \n", cfg_coordinate_format_name.c_str());

  for (const Atom &atom: motif_) {
    output->write("    %-6d %s % 3.6f % 3.6f % 3.6f\n", atom.id, material_name(atom.material).c_str(), atom.pos[0], atom.pos[1], atom.pos[2]);
  }

}

void Lattice::init_lattice_positions(
  const libconfig::Setting &material_settings,
  const libconfig::Setting &lattice_settings)
{

  lattice_map_.resize(super_cell.size[0], super_cell.size[1], super_cell.size[2], motif_.size());

  Vec3i kmesh_size = {super_cell.size[0], super_cell.size[1], super_cell.size[2]};
  if (!super_cell.periodic[0] || !super_cell.periodic[1] || !super_cell.periodic[2]) {
    output->write("\nzero padding non-periodic dimensions\n");
     // double any non-periodic dimensions for zero padding
    for (int i = 0; i < 3; ++i) {
      if (!super_cell.periodic[i]) {
        kmesh_size[i] = 2*super_cell.size[i];
      }
    }
    output->write("\npadded kspace size\n  %d  %d  %d\n", kmesh_size[0], kmesh_size[1], kmesh_size[2]);
  }

  kspace_size_ = {kmesh_size[0], kmesh_size[1], kmesh_size[2]};
  kspace_map_.resize(kspace_size_[0], kspace_size_[1], kspace_size_[2]);

// initialize everything to -1 so we can check for double assignment below

  for (int i = 0, iend = product(super_cell.size)*motif_.size(); i < iend; ++i) {
    lattice_map_[i] = -1;
  }

  for (int i = 0, iend = kspace_size_[0]*kspace_size_[1]*kspace_size_[2]; i < iend; ++i) {
    kspace_map_[i] = -1;
  }

//-----------------------------------------------------------------------------
// Generate the realspace lattice positions
//-----------------------------------------------------------------------------

  int atom_counter = 0;
  rmax_[0] = -DBL_MAX; rmax_[1] = -DBL_MAX; rmax_[2] = -DBL_MAX;
  rmin_[0] = DBL_MAX; rmin_[1] = DBL_MAX; rmin_[2] = DBL_MAX;

  Vec3i translation_vector;
  lattice_super_cell_pos_.resize(num_unit_cell_positions() * product(super_cell.size));

  // loop over the translation vectors for lattice size
  for (int i = 0; i < super_cell.size[0]; ++i) {
    for (int j = 0; j < super_cell.size[1]; ++j) {
      for (int k = 0; k < super_cell.size[2]; ++k) {

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

  output->write("\n----------------------------------------\n");
  output->write("\ncomputed lattice positions (%d)\n", atom_counter);
  for (auto i = 0; i < lattice_positions_.size(); ++i) {
    output->write("  %-6d %-6s % 3.6f % 3.6f % 3.6f | % 3.6f % 3.6f % 3.6f | %4d %4d %4d\n",
      i, lattice_materials_[i].c_str(), lattice_positions_[i][0], lattice_positions_[i][1], lattice_positions_[i][2],
                  lattice_frac_positions_[i][0], lattice_frac_positions_[i][1], lattice_frac_positions_[i][2],
      lattice_super_cell_pos_(i)[0], lattice_super_cell_pos_(i)[1], lattice_super_cell_pos_(i)[2]);

    if(!::output->is_verbose() && i > 7) {
      output->write("  ... [use verbose output for details] ... \n");
      break;
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

    output->write("  reading initial spin configuration from: %s\n", spin_filename.c_str());

    load_spin_state_from_hdf5(spin_filename);
  }

  globals::h.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);
  // globals::wij.resize(kspace_size_[0], kspace_size_[1], kspace_size_[2], 3, 3);

  std::fill(globals::h.data(), globals::h.data()+globals::num_spins3, 0.0);
  // std::fill(globals::wij.data(), globals::wij.data()+kspace_size_[0]*kspace_size_[1]*kspace_size_[2]*3*3, 0.0);

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
          Vec3 s_init = rng->sphere();
          for (auto n = 0; n < 3; ++n) {
            globals::s(i, n) = s_init[n];
          }
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

void Lattice::calc_symmetry_operations() {

  if (!symops_enabled_) {
    throw general_exception("Lattice::calc_symmetry_operations() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  output->write("\n----------------------------------------\n");
  output->write("\nsymmetry analysis\n");

  int i, j;
  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  // unit cell vectors have to be transposed because spglib wants
  // a set of 3 vectors rather than the unit cell matrix
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

  if (spglib_dataset_ == nullptr) {
    symops_enabled_ = false;
    jams_warning("spglib symmetry search failed, disabling symops");
    return;
  }

  output->write("  International\n    %s (%d)\n", spglib_dataset_->international_symbol, spglib_dataset_->spacegroup_number );
  output->write("  Hall symbol\n    %s\n", spglib_dataset_->hall_symbol );
  output->write("  Hall number\n    %d\n", spglib_dataset_->hall_number );

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  output->write("  point group\n    %s\n", ptsymbol);
  output->write("  transformation matrix\n");
  for ( i = 0; i < 3; i++ ) {
      output->write("    %f %f %f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }
  output->write("  Wyckoff letters:\n");
  for ( i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      output->write("    %c ", wl[spglib_dataset_->wyckoffs[i]]);
  }
  output->write("\n");

  output->write("  equivalent atoms:\n");
  for (i = 0; i < spglib_dataset_->n_atoms; i++) {
      output->write("    %d ", spglib_dataset_->equivalent_atoms[i]);
  }
  output->write("\n");

  output->verbose("  shifted lattice\n");
  output->verbose("    origin\n      % 3.6f % 3.6f % 3.6f\n",
    spglib_dataset_->origin_shift[0], spglib_dataset_->origin_shift[1], spglib_dataset_->origin_shift[2]);

  output->verbose("    lattice vectors\n");
  for (int i = 0; i < 3; ++i) {
    output->verbose("      % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }

  output->verbose("    positions\n");
  for (int i = 0; i < motif_.size(); ++i) {
    double bij[3];
    matmul(spglib_dataset_->transformation_matrix, spg_positions[i], bij);
    output->verbose("  %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_id_map_[spg_types[i]].name.c_str(),
      bij[0], bij[1], bij[2]);
  }

  output->write("  Standard lattice\n");
  output->write("    std lattice vectors\n");

  for (int i = 0; i < 3; ++i) {
    output->write("  % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->std_lattice[i][0], spglib_dataset_->std_lattice[i][1], spglib_dataset_->std_lattice[i][2]);
  }
  output->write("    num std atoms\n    %d\n", spglib_dataset_->n_std_atoms);

  output->write("    std_positions\n");
  for (int i = 0; i < spglib_dataset_->n_std_atoms; ++i) {
    output->write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_id_map_[spglib_dataset_->std_types[i]].name.c_str(),
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
    output->write("\n");
    output->write("unit cell is not a primitive cell\n");
    output->write("\n");
    output->write("  primitive lattice vectors:\n");

    for (int i = 0; i < 3; ++i) {
      output->write("  % 3.6f % 3.6f % 3.6f\n",
        primitive_lattice[i][0], primitive_lattice[i][1], primitive_lattice[i][2]);
    }
    output->write("\n");
    output->write("  primitive motif positions:\n");

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      output->write("  %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, material_id_map_[primitive_types[i]].name.c_str(),
        primitive_positions[i][0], primitive_positions[i][1], primitive_positions[i][2]);
      counter++;
    }
  }

  output->write("\n");
  output->write("  Symmetry operations\n");
  output->write("    num symops\n    %d\n", spglib_dataset_->n_operations);

  Mat3 rot;
  Mat3 id = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  for (int i = 0; i < spglib_dataset_->n_operations; ++i) {

    output->verbose("%d\n---\n", i);
    output->verbose("%8d  %8d  %8d\n%8d  %8d  %8d\n%8d  %8d  %8d\n",
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

Vec3 Lattice::sym_rotation(const int i, const Vec3 r_frac) const {
  assert(rotations_.size() == num_sym_ops());
  assert(i < rotations_.size() && i >= 0);
  if (symops_enabled_) {
    return rotations_[i] * r_frac;
  } else {
    return r_frac;
  }
}


// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(Vec3i& pos) const {
    for (int l = 0; l < 3; ++l) {
      if (!is_periodic(l) && (pos[l] < 0 || pos[l] >= lattice->num_unit_cells(l))) {
        return false;
      } else {
        pos[l] = (pos[l] + lattice->num_unit_cells(l))%lattice->num_unit_cells(l);
      }
    }
    return true;
}

bool Lattice::apply_boundary_conditions(int &a, int &b, int &c) const {
    if (!is_periodic(0) && (a < 0 || a >= lattice->num_unit_cells(0))) {
      return false;
    } else {
      a = (a + lattice->num_unit_cells(0))%lattice->num_unit_cells(0);
    }

    if (!is_periodic(1) && (b < 0 || b >= lattice->num_unit_cells(1))) {
      return false;
    } else {
      b = (b + lattice->num_unit_cells(1))%lattice->num_unit_cells(1);
    }

    if (!is_periodic(2) && (c < 0 || c >= lattice->num_unit_cells(2))) {
      return false;
    } else {
      c = (c + lattice->num_unit_cells(2))%lattice->num_unit_cells(2);
    }

    return true;
}

// same as the Vec3 version but accepts a Vec4 where the last component is the motif
// position difference
bool Lattice::apply_boundary_conditions(Vec4i& pos) const {
  Vec3i pos3 = {pos[0], pos[1], pos[2]};
  bool is_within_lattice = apply_boundary_conditions(pos3);
  if (is_within_lattice) {
    pos[0] = pos3[0];
    pos[1] = pos3[1];
    pos[2] = pos3[2];
  }
  return is_within_lattice;
}

double Lattice::maximum_interaction_radius() const {
  if (is_bulk_system()) {
    return rhombohedron_inradius(
            unit_cell_vector(0) * double(num_unit_cells(0)),
            unit_cell_vector(1) * double(num_unit_cells(1)),
            unit_cell_vector(2) * double(num_unit_cells(2))
    ) - 1;
  }

}

std::vector<Vec3> Lattice::generate_symmetric_points(const Vec3& r_cart, const double tolerance = 1e-6) const {

  const auto r_frac = cartesian_to_fractional(r_cart);
  std::vector<Vec3> symmetric_points;

  symmetric_points.push_back(r_cart);
  for (const auto rotation_matrix : rotations_) {
    const auto r_sym = fractional_to_cartesian(rotation_matrix * r_frac);

    if (!vec_exists_in_container(symmetric_points, r_sym, tolerance)) {
      symmetric_points.push_back(r_sym);
    }
  }

  return symmetric_points;
}

bool Lattice::is_a_symmetry_complete_set(const std::vector<Vec3> &points, const double tolerance = 1e-6) const {
  for (const auto r : points) {
    for (const auto r_sym : generate_symmetric_points(r, tolerance)) {
      if (!vec_exists_in_container(points, r_sym, tolerance)) {
        return false;
      }
    }
  }
  return true;
}