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
#include <functional>
#include <cfloat>

#include "H5Cpp.h"

#include "jams/helpers/defaults.h"
#include "jams/containers/material.h"
#include "jams/helpers/error.h"
#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/globals.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/containers/neartree.h"
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

    void output_unitcell_vectors(const Cell& cell) {
      output->write("    a = (%f, %f, %f)\n", cell.a()[0], cell.a()[1], cell.a()[2]);
      output->write("    b = (%f, %f, %f)\n", cell.b()[0], cell.b()[1], cell.b()[2]);
      output->write("    c = (%f, %f, %f)\n", cell.c()[0], cell.c()[1], cell.c()[2]);
    }
}

namespace jams {

    double llg_gyro_prefactor(const double& gyro, const double& alpha, const double& mus) {
      return -gyro /((1.0 + pow2(alpha)) * mus);
    }
}

Lattice::~Lattice() {
  if (spglib_dataset_ != nullptr) {
    spg_free_dataset(spglib_dataset_);
  }
  delete neartree_;
}

void Lattice::init_from_config(const libconfig::Config& cfg) {

  set_name("lattice");
  set_verbose(jams::config_optional<bool>(cfg.lookup("lattice"), "verbose", false));
  set_debug(jams::config_optional<bool>(cfg.lookup("lattice"), "debug", false));

  symops_enabled_ = jams::config_optional<bool>(cfg.lookup("lattice"), "symops", jams::default_lattice_symops);

  output->write("  symops %s\n", symops_enabled_ ? "true" : "false");

  read_materials_from_config(cfg.lookup("materials"));
  read_unitcell_from_config(cfg.lookup("unitcell"));
  read_lattice_from_config(cfg.lookup("lattice"));

  init_unit_cell(cfg.lookup("lattice"), cfg.lookup("unitcell"));

  if (symops_enabled_) {
    calc_symmetry_operations();
  }

  init_lattice_positions(cfg.lookup("lattice"));
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

void Lattice::read_materials_from_config(const libconfig::Setting &settings) {
  output->write("  materials\n");

  for (auto i = 0; i < settings.getLength(); ++i) {
    Material material(settings[i]);
    material.id = i;
    if (material_name_map_.insert({material.name, material}).second == false) {
      throw std::runtime_error("the material " + material.name + " is specified twice in the configuration");
    }
    material_id_map_.insert({material.id, material});
    output->write("    %-6d %s\n", material.id, material.name.c_str());
  }

  output->write("\n");

}

void Lattice::read_unitcell_from_config(const libconfig::Setting &settings) {
  // unit cell matrix is made of a,b,c lattice vectors as
  //
  // a_x  b_x  c_x
  // a_y  b_y  c_y
  // a_z  b_z  c_z
  //
  // this is consistent with the definition used by spglib
  auto basis = jams::config_required<Mat3>(settings, "basis");
  lattice_parameter  = jams::config_required<double>(settings, "parameter");

  Cell unitcell(basis);

  if (lattice_parameter < 0.0) {
    throw general_exception("lattice parameter cannot be negative", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if (lattice_parameter > 1e-7) {
    jams_warning("lattice parameter is unusually large - units should be meters");
  }

  output->write("  unit cell\n");
  output->write("    parameter %3.6e\n", lattice_parameter);
  output->write("    volume %3.6e\n", this->volume());
  output->write("\n");

  output->write("    unit cell vectors\n");
  output_unitcell_vectors(unitcell);
  output->write("\n");

  output->write("    unit cell (matrix form)\n");

  for (auto i = 0; i < 3; ++i) {
    output->write("    % 3.6f % 3.6f % 3.6f\n",
                  unitcell.matrix()[i][0], unitcell.matrix()[i][1], unitcell.matrix()[i][2]);
  }
  output->write("\n");

  output->write("    inverse unit cell (matrix form)\n");
  for (auto i = 0; i < 3; ++i) {
    output->write("    % 3.6f % 3.6f % 3.6f\n",
                  unitcell.inverse_matrix()[i][0], unitcell.inverse_matrix()[i][1], unitcell.inverse_matrix()[i][2]);
  }
  output->write("\n");
}

void Lattice::read_lattice_from_config(const libconfig::Setting &settings) {
  lattice_periodic = jams::config_optional<Vec3b>(settings, "periodic", jams::default_lattice_periodic_boundaries);
  lattice_dimensions = jams::config_required<Vec3i>(settings, "size");

  output->write("  lattice\n");
  output->write("    size %d  %d  %d\n",
                lattice_dimensions[0],lattice_dimensions[1], lattice_dimensions[2]);
  output->write("    periodic %s  %s  %s\n",
                lattice_periodic[0] ? "true" : "false",
                lattice_periodic[1] ? "true" : "false",
                lattice_periodic[2] ? "true" : "false");
  output->write("\n");
}

void Lattice::init_unit_cell(const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings) {
  using namespace globals;
  using std::string;

  supercell = scale(unitcell, lattice_dimensions);

  if (lattice_settings.exists("global_rotation") && lattice_settings.exists("orientation_axis")) {
    jams_warning("Orientation and global rotation are both in config. Orientation will be applied first and then global rotation.");
  }

  if (lattice_settings.exists("orientation_axis")) {
    auto reference_axis = jams::config_required<Vec3>(lattice_settings, "orientation_axis");
    auto lattice_vector = jams::config_required<Vec3>(lattice_settings, "orientation_lattice_vector");

    global_reorientation(reference_axis, lattice_vector);
  }

  if (lattice_settings.exists("global_rotation")) {
    global_rotation(jams::config_optional(lattice_settings, "global_rotation", kIdentityMat3));
  }

  CoordinateFormat cfg_coordinate_format = CoordinateFormat::Fractional;

  std::string cfg_coordinate_format_name = jams::config_optional<string>(unitcell_settings, "coordinate_format", "FRACTIONAL");

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

  output->write("  motif positions (%s)\n", position_filename.c_str());

  output->write("  format \n", cfg_coordinate_format_name.c_str());

  for (const Atom &atom: motif_) {
    output->write("    %-6d %s % 3.6f % 3.6f % 3.6f\n", atom.id, material_name(atom.material).c_str(), atom.pos[0], atom.pos[1], atom.pos[2]);
  }
  output->write("\n");

}

void Lattice::global_rotation(const Mat3& rotation_matrix) {
  auto volume_before = ::volume(unitcell);

  unitcell = rotate(unitcell, rotation_matrix);
  supercell = rotate(supercell, rotation_matrix);

  auto volume_after = ::volume(unitcell);

  if (std::abs(volume_before - volume_after) > 1e-6) {
    jams_error("unitcell volume has changed after rotation");
  }

  output->write("  global rotated lattice vectors\n");
  output_unitcell_vectors(unitcell);
  output->write("\n");
}

void Lattice::global_reorientation(const Vec3 &reference, const Vec3 &vector) {

  Vec3 orientation_cartesian_vector = normalize(unitcell.matrix() * vector);

  output->write("  orientation_axis\n");
  output->write("    % 3.6f % 3.6f % 3.6f\n", reference[0], reference[1], reference[2]);

  output->write("  orientation_lattice_vector\n");
  output->write("    % 3.6f % 3.6f % 3.6f\n", vector[0], vector[1], vector[2]);

  output->write("  orientation_cartesian_vector\n");
  output->write("    % 3.6f % 3.6f % 3.6f\n", orientation_cartesian_vector[0], orientation_cartesian_vector[1],
                orientation_cartesian_vector[2]);

  Mat3 orientation_matrix = rotation_matrix_between_vectors(orientation_cartesian_vector, reference);

  output->write("  orientation rotation matrix\n");
  output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[0][0], orientation_matrix[0][1],
                orientation_matrix[0][2]);
  output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[1][0], orientation_matrix[1][1],
                orientation_matrix[1][2]);
  output->write("    % 8.8f, % 8.8f, % 8.8f\n", orientation_matrix[2][0], orientation_matrix[2][1],
                orientation_matrix[2][2]);
  output->write("\n");

  Vec3 rotated_orientation_vector = orientation_matrix * orientation_cartesian_vector;

  if (verbose_is_enabled()) {
    output->write("  rotated_orientation_vector\n");
    output->write("    % 3.6f % 3.6f % 3.6f\n", rotated_orientation_vector[0], rotated_orientation_vector[1],
            rotated_orientation_vector[2]);
  }

  auto volume_before = ::volume(unitcell);
  rotate(unitcell, orientation_matrix);
  rotate(supercell, orientation_matrix);
  auto volume_after = ::volume(unitcell);

  if (std::abs(volume_before - volume_after) > 1e-6) {
    jams_error("unitcell volume has changed after rotation");
  }

  output->write("  oriented lattice vectors\n");
  output_unitcell_vectors(unitcell);
  output->write("\n");
}


void Lattice::init_lattice_positions(const libconfig::Setting &lattice_settings)
{

  lattice_map_.resize(lattice_dimensions[0], lattice_dimensions[1], lattice_dimensions[2], motif_.size());

  Vec3i kmesh_size = {lattice_dimensions[0], lattice_dimensions[1], lattice_dimensions[2]};
  if (!lattice_periodic[0] || !lattice_periodic[1] || !lattice_periodic[2]) {
    output->write("\nzero padding non-periodic dimensions\n");
    // double any non-periodic dimensions for zero padding
    for (int i = 0; i < 3; ++i) {
      if (!lattice_periodic[i]) {
        kmesh_size[i] = 2*lattice_dimensions[i];
      }
    }
    output->write("\npadded kspace size\n  %d  %d  %d\n", kmesh_size[0], kmesh_size[1], kmesh_size[2]);
  }

  kspace_size_ = {kmesh_size[0], kmesh_size[1], kmesh_size[2]};
  kspace_map_.resize(kspace_size_[0], kspace_size_[1], kspace_size_[2]);

// initialize everything to -1 so we can check for double assignment below

  for (int i = 0, iend = product(lattice_dimensions)*motif_.size(); i < iend; ++i) {
    lattice_map_[i] = -1;
  }

  for (int i = 0, iend = kspace_size_[0]*kspace_size_[1]*kspace_size_[2]; i < iend; ++i) {
    kspace_map_[i] = -1;
  }

// Generate the realspace lattice positions

  int atom_counter = 0;
  rmax_[0] = -DBL_MAX; rmax_[1] = -DBL_MAX; rmax_[2] = -DBL_MAX;

  lattice_super_cell_pos_.resize(num_motif_positions() * product(lattice_dimensions));

  // loop over the translation vectors for lattice size
  for (auto i = 0; i < lattice_dimensions[0]; ++i) {
    for (auto j = 0; j < lattice_dimensions[1]; ++j) {
      for (auto k = 0; k < lattice_dimensions[2]; ++k) {
        for (auto m = 0; m < motif_.size(); ++m) {
          Vec3i translation_vector = {i, j, k};

          // number the site in the fast integer lattice
          lattice_map_(i, j, k, m) = atom_counter;

          lattice_super_cell_pos_(atom_counter) = translation_vector;
          lattice_positions_.push_back(generate_position(motif_[m].pos, translation_vector));
          lattice_materials_.push_back(material_id_map_[motif_[m].material].name);

          atoms_.push_back({atom_counter, motif_[m].material, generate_position(motif_[m].pos, translation_vector)});
          atom_counter++;

          // store max coordinates
          for (int n = 0; n < 3; ++n) {
            if (lattice_positions_.back()[n] > rmax_[n]) {
              rmax_[n] = lattice_positions_.back()[n];
            }
          }

        }
      }
    }
  }

  if (atom_counter == 0) {
    jams_error("the number of computed lattice sites was zero, check input");
  }

  Cell neartree_cell = supercell;
  auto distance_metric = [neartree_cell](const Atom& a, const Atom& b)->double {
      return abs(::minimum_image(neartree_cell, a.pos, b.pos));
  };

  neartree_ = new NearTree<Atom, NeartreeFunctorType>(distance_metric, atoms_);

  globals::num_spins = atom_counter;
  globals::num_spins3 = 3*atom_counter;

  output->write("  computed lattice positions %d\n", atom_counter);
  for (auto i = 0; i < lattice_positions_.size(); ++i) {
    output->write("    %-6d %-6s % 3.6f % 3.6f % 3.6f | %4d %4d %4d\n",
                  i, lattice_materials_[i].c_str(),
                  lattice_positions_[i][0], lattice_positions_[i][1], lattice_positions_[i][2],
                  lattice_super_cell_pos_(i)[0], lattice_super_cell_pos_(i)[1], lattice_super_cell_pos_(i)[2]);

    if(!verbose_is_enabled() && i > 7) {
      output->write("    ... [use verbose output for details] ... \n");
      break;
    }
  }


// initialize global arrays
  globals::s.resize(globals::num_spins, 3);
  globals::ds_dt.resize(globals::num_spins, 3);
  globals::h.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);

  globals::h.zero();

  num_of_material_.resize(num_materials(), 0);

  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto material = material_name_map_[lattice_materials_[i]];

    globals::mus(i)   = material.moment;
    globals::alpha(i) = material.alpha;
    globals::gyro(i)  = jams::llg_gyro_prefactor(material.gyro, material.alpha, material.moment);

    for (auto n = 0; n < 3; ++n) {
      globals::s(i, n) = material.spin[n];
    }

    if (material.randomize) {
      Vec3 s_init = rng->sphere();
      for (auto n = 0; n < 3; ++n) {
        globals::s(i, n) = s_init[n];
      }
    }

    num_of_material_[material.id]++;
  }

  bool initial_spin_state_is_a_file = lattice_settings.exists("spins");

  if (initial_spin_state_is_a_file) {
    std::string spin_filename = lattice_settings["spins"];

    output->write("  reading initial spin configuration from: %s\n", spin_filename.c_str());

    load_spin_state_from_hdf5(spin_filename);
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

Vec3 Lattice::generate_position(
        const Vec3 unit_cell_frac_pos,
        const Vec3i translation_vector) const
{
  return unitcell.matrix() * generate_fractional_position(unit_cell_frac_pos, translation_vector);
}

// generate a position within a periodic image of the entire system
Vec3 Lattice::generate_image_position(
        const Vec3 unit_cell_cart_pos,
        const Vec3i image_vector) const
{
  Vec3 frac_pos = cartesian_to_fractional(unit_cell_cart_pos);
  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      frac_pos[n] = frac_pos[n] + image_vector[n] * lattice_dimensions[n];
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

void Lattice::calc_symmetry_operations() {

  if (!symops_enabled_) {
    throw general_exception("Lattice::calc_symmetry_operations() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  output->write("  symmetry analysis\n");

  int i, j;
  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  // unit cell vectors have to be transposed because spglib wants
  // a set of 3 vectors rather than the unit cell matrix
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      spg_lattice[i][j] = unitcell.matrix()[i][j];
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

  output->write("    International %s (%d)\n", spglib_dataset_->international_symbol, spglib_dataset_->spacegroup_number );
  output->write("    Hall symbol %s\n", spglib_dataset_->hall_symbol );
  output->write("    Hall number %d\n", spglib_dataset_->hall_number );

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  output->write("    point group  %s\n", ptsymbol);
  output->write("    transformation matrix\n");
  for ( i = 0; i < 3; i++ ) {
      output->write("    %f %f %f\n",
      spglib_dataset_->transformation_matrix[i][0],
      spglib_dataset_->transformation_matrix[i][1],
      spglib_dataset_->transformation_matrix[i][2]);
  }
  output->write("    Wyckoff letters ");
  for ( i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      output->write("%c ", wl[spglib_dataset_->wyckoffs[i]]);
  }
  output->write("\n");

  output->write("    equivalent atoms ");
  for (i = 0; i < spglib_dataset_->n_atoms; i++) {
      output->write("%d ", spglib_dataset_->equivalent_atoms[i]);
  }
  output->write("\n");

  if (verbose_is_enabled()) {
    output->verbose("    shifted lattice\n");
    output->verbose("    origin % 3.6f % 3.6f % 3.6f\n",
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
  }

  output->write("    standard lattice\n");
  output->write("    std lattice vectors\n");

  for (int i = 0; i < 3; ++i) {
    output->write("    % 3.6f % 3.6f % 3.6f\n",
      spglib_dataset_->std_lattice[i][0], spglib_dataset_->std_lattice[i][1], spglib_dataset_->std_lattice[i][2]);
  }
  output->write("    num std atoms %d\n", spglib_dataset_->n_std_atoms);

  output->write("    std_positions\n");
  for (int i = 0; i < spglib_dataset_->n_std_atoms; ++i) {
    output->write("    %-6d %s % 3.6f % 3.6f % 3.6f\n", i, material_id_map_[spglib_dataset_->std_types[i]].name.c_str(),
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
    output->write("    !!! unit cell is not a primitive cell !!!\n");
    output->write("\n");
    output->write("    primitive lattice vectors:\n");

    for (int i = 0; i < 3; ++i) {
      output->write("    % 3.6f % 3.6f % 3.6f\n",
        primitive_lattice[i][0], primitive_lattice[i][1], primitive_lattice[i][2]);
    }
    output->write("\n");
    output->write("    primitive motif positions:\n");

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      output->write("    %-6d %s % 3.6f % 3.6f % 3.6f\n", counter, material_id_map_[primitive_types[i]].name.c_str(),
        primitive_positions[i][0], primitive_positions[i][1], primitive_positions[i][2]);
      counter++;
    }
  }
  output->write("\n");
  output->write("    num symops %d\n", spglib_dataset_->n_operations);

  Mat3 rot;
  Mat3 id = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  for (int i = 0; i < spglib_dataset_->n_operations; ++i) {

    if (verbose_is_enabled()) {
      output->verbose("    %d\n---\n", i);
      output->verbose("    %8d  %8d  %8d\n%8d  %8d  %8d\n%8d  %8d  %8d\n",
              spglib_dataset_->rotations[i][0][0], spglib_dataset_->rotations[i][0][1],
              spglib_dataset_->rotations[i][0][2],
              spglib_dataset_->rotations[i][1][0], spglib_dataset_->rotations[i][1][1],
              spglib_dataset_->rotations[i][1][2],
              spglib_dataset_->rotations[i][2][0], spglib_dataset_->rotations[i][2][1],
              spglib_dataset_->rotations[i][2][2]);
    }

    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        rot[m][n] = spglib_dataset_->rotations[i][m][n];
      }
    }

    rotations_.push_back(rot);
  }
  output->write("\n");

}


// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(Vec3i& pos) const {
    for (int l = 0; l < 3; ++l) {
      if (!is_periodic(l) && (pos[l] < 0 || pos[l] >= lattice->size(l))) {
        return false;
      } else {
        pos[l] = (pos[l] + lattice->size(l))%lattice->size(l);
      }
    }
    return true;
}

bool Lattice::apply_boundary_conditions(int &a, int &b, int &c) const {
    if (!is_periodic(0) && (a < 0 || a >= lattice->size(0))) {
      return false;
    } else {
      a = (a + lattice->size(0))%lattice->size(0);
    }

    if (!is_periodic(1) && (b < 0 || b >= lattice->size(1))) {
      return false;
    } else {
      b = (b + lattice->size(1))%lattice->size(1);
    }

    if (!is_periodic(2) && (c < 0 || c >= lattice->size(2))) {
      return false;
    } else {
      c = (c + lattice->size(2))%lattice->size(2);
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

double Lattice::max_interaction_radius() const {
  if (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2)) {
    return rhombohedron_inradius(supercell.a(), supercell.b(), supercell.c()) - 1;
  }
  return 0.0;
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


