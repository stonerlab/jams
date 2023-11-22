// Copyright 2014 Joseph Barker. All rights reserved.

extern "C"{
    #include <spglib.h>
}

#include "jams/core/lattice.h"

#include <libconfig.h++>
#include <cstddef>

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include <cmath>
#include <pcg_random.hpp>

#include <jams/common.h>
#include "jams/helpers/defaults.h"
#include "jams/containers/material.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/core/globals.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/containers/neartree.h"
#include "jams/interface/highfive.h"
#include "jams/helpers/load.h"
#include "jams/helpers/interaction_calculator.h"
#include "lattice.h"
#include <jams/maths/parallelepiped.h>
#include <jams/lattice/minimum_image.h>


using std::cout;
using std::endl;
using libconfig::Setting;
using libconfig::Config;

namespace {
    Vec3 shift_fractional_coordinate_to_zero_one(Vec3 r, const double eps = jams::defaults::lattice_tolerance) {
      for (auto n = 0; n < 3; ++n) {
        if (r[n] < 0.0) {
          r[n] = r[n] + 1.0;
        }
        // If we end up exactly on the opposite face/edge of the cell then
        // this should actually be in the next cell (i.e. fractional coordinates
        // are in the range 0 <= r[n] < 1. So we must map coordinates equal to 1
        // back to 0.
        if (approximately_equal(r[n], 1.0, eps)) {
          r[n] = 0.0;
        }
      }
      return r;
    }

    bool is_fractional_coordinate_valid(const Vec3 &r, const double eps = jams::defaults::lattice_tolerance) {
      // check fractional coordinates are in the range 0 <= r[n] < 1
      for (auto n = 0; n < 3; ++n) {
        if (r[n] < 0.0 || r[n] > 1.0 || approximately_equal(r[n], 1.0, eps)) {
          return false;
        }
      }
      return true;
    }

    void output_unitcell_vectors(const Cell& cell) {
      cout << "    a = " << jams::fmt::decimal << cell.a() << "\n";
      cout << "    b = " << jams::fmt::decimal << cell.b() << "\n";
      cout << "    c = " << jams::fmt::decimal << cell.c() << "\n";
    }

    void output_unitcell_inverse_vectors(const Cell& cell) {
      cout << "    a_inv = " << jams::fmt::decimal << cell.a_inv() << "\n";
      cout << "    b_inv = " << jams::fmt::decimal << cell.b_inv() << "\n";
      cout << "    c_inv = " << jams::fmt::decimal << cell.c_inv() << "\n";
    }
}

namespace jams {
    double landau_lifshitz_gyro_prefactor(const double& gyro, const double& alpha, const double& mus) {
      return gyro;
    }

    double gilbert_gyro_prefactor(const double& gyro, const double& alpha, const double& mus) {
      return gyro /(1.0 + pow2(alpha));
    }
}

Lattice::~Lattice() {
  if (spglib_dataset_ != nullptr) {
    spg_free_dataset(spglib_dataset_);
  }
}


double Lattice::parameter() const {
  return lattice_parameter;
}


int Lattice::size(int i) const {
  return lattice_dimensions[i];
}

Vec3i Lattice::size() const {
  return lattice_dimensions;
}

int Lattice::num_motif_atoms() const {
  return motif_.size();
}

Vec3 Lattice::a() const {
  return unitcell.a();
}

Vec3 Lattice::b() const {
  return unitcell.b();
}

Vec3 Lattice::c() const {
  return unitcell.c();
}

int
Lattice::num_materials() const {
  return materials_.size();
}

std::string
Lattice::material_name(int uid) const {
  return materials_.name(uid);
}

int
Lattice::material_id(const std::string &name) const {
  return materials_.id(name);
}

int
Lattice::atom_material_id(const int &i) const {
  assert(i < atoms_.size());
  return atoms_[i].material_index;
}

std::string
Lattice::atom_material_name(const int &i) const {
  assert(i < atoms_.size());
  return material_name(atom_material_id(i));
}

const Vec3 &
Lattice::atom_position(const int &i) const {
  return cartesian_positions_[i];
}

Vec3
Lattice::displacement(const Vec3 &r_i, const Vec3 &r_j) const {
  return jams::minimum_image(supercell.a(), supercell.b(), supercell.c(), supercell.periodic(), r_i, r_j, jams::defaults::lattice_tolerance);
}

Vec3 Lattice::displacement(const unsigned &i, const unsigned &j) const {
  return jams::minimum_image(supercell.a(), supercell.b(), supercell.c(), supercell.periodic(), atoms_[i].position, atoms_[j].position, jams::defaults::lattice_tolerance);
}

Vec3
Lattice::cartesian_to_fractional(const Vec3 &r_cart) const {
  return unitcell.inverse_matrix() * r_cart;
}

Vec3
Lattice::fractional_to_cartesian(const Vec3 &r_frac) const {
  return unitcell.matrix() * r_frac;
}

const Vec3 &
Lattice::rmax() const {
  return rmax_;
};

int Lattice::site_index_by_unit_cell(const int &i, const int &j, const int &k, const int &m) const {
  assert(i < lattice_dimensions[0]);
  assert(i >= 0);
  assert(j < lattice_dimensions[1]);
  assert(j >= 0);
  assert(k < lattice_dimensions[2]);
  assert(k >= 0);
  assert(m < num_motif_atoms());
  assert(m >= 0);

  return lattice_map_(i, j, k, m);
}

bool Lattice::is_periodic(int i) const {
  return lattice_periodic[i];
}

const Vec3b & Lattice::periodic_boundaries() const {
  return lattice_periodic;
}

const Vec3i &Lattice::kspace_size() const {
  return kspace_size_;
}

void Lattice::init_from_config(const libconfig::Config& cfg) {

  set_name("lattice");
  set_verbose(jams::config_optional<bool>(cfg.lookup("lattice"), "verbose", false));
  set_debug(jams::config_optional<bool>(cfg.lookup("lattice"), "debug", false));

  symops_enabled_ = jams::config_optional<bool>(cfg.lookup("unitcell"), "symops", jams::defaults::unitcell_symops);

  cout << "  symops " << symops_enabled_ << "\n";

  read_materials_from_config(cfg.lookup("materials"));
  read_unitcell_from_config(cfg.lookup("unitcell"));
  read_lattice_from_config(cfg.lookup("lattice"));

  init_unit_cell(cfg.lookup("lattice"), cfg.lookup("unitcell"));

  double interaction_calculator_radius = 0.0;
  interaction_calculator_radius = jams::config_optional<double>(cfg.lookup("unitcell"), "calculate_interaction_vectors", interaction_calculator_radius);

  if (interaction_calculator_radius != 0.0) {
    cout << "calculating interaction vectors up to r = " << interaction_calculator_radius << std::endl;
    jams::interaction_calculator(unitcell, motif_, interaction_calculator_radius);
  }

  if (symops_enabled_) {

    if (motif_.size() > jams::defaults::warning_unitcell_symops_size) {
      jams_warning("symmetry calculation may be slow as unit cell has more than %d atoms and symops is turned on", jams::defaults::warning_unitcell_symops_size);
    }

    calc_symmetry_operations();
  }

  generate_supercell(cfg.lookup("lattice"));
}

void Lattice::read_motif_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format) {
  Atom atom;
  std::string atom_name;

  motif_.clear();

  for (int i = 0; i < positions.getLength(); ++i) {
    atom_name = positions[i][0].c_str();

    // check the material type is defined
    if (!materials_.contains(atom_name)) {
      throw jams::runtime_error("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material_index = materials_.id(atom_name);

    atom.position[0] = positions[i][1][0];
    atom.position[1] = positions[i][1][1];
    atom.position[2] = positions[i][1][2];

    if (coordinate_format == CoordinateFormat::CARTESIAN) {
      atom.position = cartesian_to_fractional(atom.position);
    }

    atom.position = shift_fractional_coordinate_to_zero_one(atom.position);

    if (!is_fractional_coordinate_valid(atom.position)) {
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
    throw jams::runtime_error("failed to open position file " + filename, __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  motif_.clear();

  // read the motif into an array from the positions file
  while (getline(position_file, line)) {
    if(string_is_comment(line)) {
      continue;
    }
    std::stringstream line_as_stream;
    std::string atom_name;
    Atom atom;

    line_as_stream.str(line);

    // read atom type name
    line_as_stream >> atom_name >> atom.position[0] >> atom.position[1] >> atom.position[2];

    if (coordinate_format == CoordinateFormat::CARTESIAN) {
      atom.position = cartesian_to_fractional(atom.position);
    }

    atom.position = shift_fractional_coordinate_to_zero_one(atom.position);

    if (!is_fractional_coordinate_valid(atom.position)) {
      throw std::runtime_error("atom position " + std::to_string(motif_.size()) + " is not a valid fractional coordinate");
    }
    // check the material type is defined
    if (!materials_.contains(atom_name)) {
      throw jams::runtime_error("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material_index = materials_.id(atom_name);
    atom.id = motif_.size();

    motif_.push_back(atom);
  }
  position_file.close();
}

void Lattice::read_materials_from_config(const libconfig::Setting &settings) {
  cout << "  materials\n";

  for (auto i = 0; i < settings.getLength(); ++i) {
    Material material(settings[i]);

    if (materials_.contains(material.name)) {
      throw std::runtime_error("the material " + material.name + " is specified twice in the configuration");
    }

    materials_.insert(material.name, material);

    cout << "    " << i << " " << material.name << "\n";
  }

  cout << "\n";

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

  unitcell = Cell(basis);

  if (lattice_parameter < 0.0) {
    throw jams::runtime_error("lattice parameter cannot be negative", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if (lattice_parameter == 0.0) {
    throw jams::runtime_error("lattice parameter cannot be zero", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if (lattice_parameter < 1e-10) {
    jams_warning("lattice parameter is unusually small - units should be meters");
  }

  if (lattice_parameter > 1e-7) {
    jams_warning("lattice parameter is unusually large - units should be meters");
  }

  cout << "  unit cell\n";
  cout << "    parameter " << jams::fmt::sci << lattice_parameter << " (m)\n";
  cout << "    volume " << jams::fmt::sci << ::volume(unitcell) * pow3(lattice_parameter) << " (m^3)\n";
  cout << "\n";

  cout << "    unit cell vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";

  cout << "    unit cell (matrix form)\n";

  for (auto i = 0; i < 3; ++i) {
    cout << "    " << jams::fmt::decimal << unitcell.matrix()[i] << "\n";
  }
  cout << "\n";

  cout << "    unit cell inverse vectors\n";
  output_unitcell_inverse_vectors(unitcell);
  cout << "\n";

  cout << "    inverse unit cell (matrix form)\n";
  for (auto i = 0; i < 3; ++i) {
    cout << "    " << jams::fmt::decimal << unitcell.inverse_matrix()[i] << "\n";
  }
  cout << "\n";
}

void Lattice::read_lattice_from_config(const libconfig::Setting &settings) {
  lattice_periodic = jams::config_optional<Vec3b>(settings, "periodic", jams::defaults::lattice_periodic_boundaries);
  lattice_dimensions = jams::config_required<Vec3i>(settings, "size");

  supercell = scale(Cell(unitcell.matrix(), lattice_periodic), lattice_dimensions);

  cout << "  lattice\n";
  cout << "    size " << lattice_dimensions << " (unit cells)\n";
  cout << "    periodic " << lattice_periodic << "\n";
  cout << "    volume " << jams::fmt::sci << ::volume(supercell) * pow3(lattice_parameter) << "\n";
  cout << "\n";

  if(settings.exists("impurities")) {
    impurity_map_ = read_impurities_from_config(settings["impurities"]);
    impurity_seed_ = jams::config_optional<unsigned>(settings, "impurities_seed", jams::instance().random_generator()());
    cout << jams::fmt::integer << "  impurity seed " << impurity_seed_ << "\n";
  }
}

void Lattice::init_unit_cell(const libconfig::Setting &lattice_settings, const libconfig::Setting &unitcell_settings) {
  using std::string;

  if (lattice_settings.exists("global_rotation") && lattice_settings.exists("orientation_axis")) {
    jams_warning("Orientation and global rotation are both in config. Orientation will be applied first and then global rotation.");
  }

  if (lattice_settings.exists("orientation_axis")) {
    if (lattice_settings.exists("orientation_lattice_vector") && lattice_settings.exists("orientation_cartesian_vector")) {
      jams_die("Only one of 'orientation_lattice_vector' or 'orientation_cartesian_vector' can be defined");
    }
    auto reference_axis = jams::config_required<Vec3>(lattice_settings, "orientation_axis");
    if (lattice_settings.exists("orientation_lattice_vector")) {
      auto lattice_vector = jams::config_required<Vec3>(lattice_settings, "orientation_lattice_vector");
      global_reorientation(reference_axis, lattice_vector);
    } else if (lattice_settings.exists("orientation_cartesian_vector")) {
      auto lattice_vector = jams::config_required<Vec3>(lattice_settings, "orientation_cartesian_vector");
      global_reorientation(reference_axis, cartesian_to_fractional(lattice_vector));
    }
  }

  if (lattice_settings.exists("global_rotation")) {
    global_rotation(jams::config_optional(lattice_settings, "global_rotation", kIdentityMat3));
  }

  CoordinateFormat cfg_coordinate_format = CoordinateFormat::FRACTIONAL;

  std::string cfg_coordinate_format_name = jams::config_optional<string>(unitcell_settings, "coordinate_format", "FRACTIONAL");

  if (capitalize(cfg_coordinate_format_name) == "FRACTIONAL") {
    cfg_coordinate_format = CoordinateFormat::FRACTIONAL;
  } else if (capitalize(cfg_coordinate_format_name) == "CARTESIAN") {
    cfg_coordinate_format = CoordinateFormat::CARTESIAN;
  } else {
    throw std::runtime_error("Unknown coordinate format for atom positions in unit cell");
  }

  std::string position_filename;
  if (unitcell_settings["positions"].isList()) {
    position_filename = globals::simulation_name + ".cfg";
    read_motif_from_config(unitcell_settings["positions"], cfg_coordinate_format);
  } else {
    position_filename = unitcell_settings["positions"].c_str();
    read_motif_from_file(position_filename, cfg_coordinate_format);
  }

  cout << "  motif positions " << position_filename << "\n";
  cout << "  format " << cfg_coordinate_format_name << "\n";

  for (const Atom &atom: motif_) {
    cout << "    " << jams::fmt::integer << atom.id << " ";
    cout << materials_.name(atom.material_index) << " ";
    cout << jams::fmt::decimal << atom.position << "\n";
  }
  cout << endl;

  bool check_closeness = jams::config_optional<bool>(unitcell_settings, "check_closeness", true);

  if (check_closeness) {
    cout << "checking no atoms are too close together..." << std::flush;

    for (auto i = 0; i < motif_.size(); ++i) {
      for (auto j = i + 1; j < motif_.size(); ++j) {
        auto distance = norm(
            jams::minimum_image(unitcell.a(), unitcell.b(), unitcell.c(),
                                unitcell.periodic(),
                                fractional_to_cartesian(motif_[i].position),
                                fractional_to_cartesian(motif_[j].position),
                                jams::defaults::lattice_tolerance));
        if (distance < jams::defaults::lattice_tolerance) {
          throw std::runtime_error(
              "motif positions " + std::to_string(i) + " and " +
              std::to_string(j) +
              " are closer than the default lattice tolerance");
        }
      }
    }
    cout << "ok" << endl;
  }



}

void Lattice::global_rotation(const Mat3& rotation_matrix) {
  auto volume_before = ::volume(unitcell);

  unitcell = rotate(unitcell, rotation_matrix);
  supercell = rotate(supercell, rotation_matrix);

  auto volume_after = ::volume(unitcell);

  if (!approximately_equal(volume_before, volume_after, pow3(jams::defaults::lattice_tolerance))) {
    jams_die("unitcell volume has changed after rotation");
  }

  cout << "  global rotated lattice vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";
  cout << "  global rotated inverse vectors\n";
  output_unitcell_inverse_vectors(unitcell);
  cout << "\n";
}

void Lattice::global_reorientation(const Vec3 &reference, const Vec3 &vector) {

  Vec3 orientation_cartesian_vector = normalize(unitcell.matrix() * vector);

  cout << "  orientation_axis " << reference << "\n";
  cout << "  orientation_lattice_vector " << vector << "\n";
  cout << "  orientation_cartesian_vector " << orientation_cartesian_vector << "\n";

  global_orientation_matrix_ = rotation_matrix_between_vectors(orientation_cartesian_vector, reference);

  cout << "  orientation rotation matrix \n";
  cout << "    " << global_orientation_matrix_[0] << "\n";
  cout << "    " << global_orientation_matrix_[1] << "\n";
  cout << "    " << global_orientation_matrix_[2] << "\n";
  cout << "\n";

  Vec3 rotated_orientation_vector = global_orientation_matrix_ * orientation_cartesian_vector;

  if (verbose_is_enabled()) {
    cout << "  rotated_orientation_vector\n";
    cout << "    " << rotated_orientation_vector << "\n";
  }

  auto volume_before = ::volume(unitcell);
  unitcell = rotate(unitcell, global_orientation_matrix_);
  supercell = rotate(supercell, global_orientation_matrix_);
  auto volume_after = ::volume(unitcell);

  if (!approximately_equal(volume_before, volume_after, pow3(jams::defaults::lattice_tolerance))) {
    jams_die("unitcell volume has changed after rotation");
  }

  cout << "  oriented lattice vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";

  cout << "  oriented inverse vectors\n";
  output_unitcell_inverse_vectors(unitcell);
  cout << "\n";
}


void Lattice::generate_supercell(const libconfig::Setting &lattice_settings)
{
  Vec3i kmesh_size = {lattice_dimensions[0], lattice_dimensions[1], lattice_dimensions[2]};

  if (!lattice_periodic[0] || !lattice_periodic[1] || !lattice_periodic[2]) {
    cout << "\nzero padding non-periodic dimensions\n";
    // double any non-periodic dimensions for zero padding
    for (auto i = 0; i < 3; ++i) {
      if (!lattice_periodic[i]) {
        kmesh_size[i] = 2*lattice_dimensions[i];
      }
    }
    cout << "\npadded kspace size " << kmesh_size << "\n";
  }

  kspace_size_ = {kmesh_size[0], kmesh_size[1], kmesh_size[2]};
  kspace_map_.resize(kspace_size_[0], kspace_size_[1], kspace_size_[2]);
  kspace_map_.fill(-1);



  cout << "\nkspace size " << kmesh_size << "\n";



  lattice_map_.resize(this->size(0), this->size(1), this->size(2), this->num_motif_atoms());
  // initialize everything to -1 so we can check for double assignment below
  lattice_map_.fill(-1);

  const auto num_cells = product(lattice_dimensions);
  const auto expected_num_atoms = num_motif_atoms() * num_cells;

  cell_centers_.reserve(num_cells);
  cell_offsets_.reserve(num_cells);
  atoms_.reserve(expected_num_atoms);
  atom_to_cell_lookup_.reserve(expected_num_atoms);

  auto impurity_rand = std::bind(std::uniform_real_distribution<>(), pcg32(impurity_seed_));

  // loop over the translation vectors for lattice size
  int atom_counter = 0;
  std::vector<size_t> type_counter(materials_.size(), 0);


  unsigned cell_counter = 0;
  for (auto i = 0; i < lattice_dimensions[0]; ++i) {
    for (auto j = 0; j < lattice_dimensions[1]; ++j) {
      for (auto k = 0; k < lattice_dimensions[2]; ++k) {
        auto cell_offset = Vec3i{{i, j, k}};
        cell_offsets_.push_back(cell_offset);
        cell_centers_.push_back(generate_cartesian_lattice_position_from_fractional(Vec3{0.5,0.5,0.5}, cell_offset));

        for (auto m = 0; m < motif_.size(); ++m) {
          auto position    = generate_cartesian_lattice_position_from_fractional(motif_[m].position, cell_offset);
          auto material    = motif_[m].material_index;

          if (impurity_map_.count(material)) {
            auto impurity    = impurity_map_[material];

            if (impurity_rand() < impurity.fraction) {
              material = impurity.material;
            }
          }

          atoms_.push_back({atom_counter, material, m, position});

          cartesian_positions_.push_back(position);
          fractional_positions_.push_back(cartesian_to_fractional(position));

          atom_to_cell_lookup_.push_back(cell_counter);

          // number the site in the fast integer lattice
          lattice_map_(i, j, k, m) = atom_counter;

          type_counter[material]++;
          atom_counter++;
        }
        cell_counter++;
      }
    }
  }

  if (atom_counter == 0) {
    jams_die("the number of computed lattice sites was zero, check input");
  }

  cout << "    lattice material count\n";
  for (auto n = 0; n < type_counter.size(); ++n) {
    cout << "      " << materials_.name(n) << ": " << type_counter[n] << "\n";
  }

  // this is the top right hand corner of the top right unit cell in the super cell
  rmax_ = generate_cartesian_lattice_position_from_fractional(Vec3{0.0, 0.0, 0.0}, lattice_dimensions);

  if (atom_counter == 0) {
    jams_die("the number of computed lattice sites was zero, check input");
  }

  globals::num_spins = atom_counter;
  globals::num_spins3 = 3*atom_counter;

  cout << "  computed lattice positions " << atom_counter << "\n";
  for (auto i = 0; i < atoms_.size(); ++i) {
    cout << "    " << jams::fmt::fixed_integer << i << " ";
    cout << std::setw(8) << materials_.name(atoms_[i].material_index) << " ";
    cout << jams::fmt::decimal << atoms_[i].position << " ";
    cout << jams::fmt::fixed_integer << cell_offset(i) << "\n";
    if(!verbose_is_enabled() && i > 7) {
      cout << "    ... [use verbose output for details] ... \n";
      break;
    }
  }


// initialize global arrays
  globals::s.resize(globals::num_spins, 3);
  globals::ds_dt.resize(globals::num_spins, 3);
  globals::h.resize(globals::num_spins, 3);
  globals::positions.resize(globals::num_spins, 3);
  globals::alpha.resize(globals::num_spins);
  globals::mus.resize(globals::num_spins);
  globals::gyro.resize(globals::num_spins);

  bool use_gilbert_prefactor = jams::config_optional<bool>(
      globals::config->lookup("solver"), "gilbert_prefactor", false);

  bool normalise_spins = jams::config_optional<bool>(
      lattice_settings, "normalise_spins", true);

  pcg32 rng = pcg_extras::seed_seq_from<std::random_device>();
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto material = materials_[atom_material_id(i)];

    globals::mus(i)   = material.moment;
    globals::alpha(i) = material.alpha;

    if (use_gilbert_prefactor) {
      globals::gyro(i)  = jams::gilbert_gyro_prefactor(material.gyro, material.alpha, material.moment);
    } else {
      globals::gyro(i) = jams::landau_lifshitz_gyro_prefactor(material.gyro, material.alpha, material.moment);
    }

    Vec3 spin = material.spin;

    if (material.randomize) {
      spin = jams::uniform_random_sphere(rng);
    }

    // lattice vacancies have a moment of zero and a spin vector of zero
    if (material.moment == 0.0) {
      spin = Vec3{0.0, 0.0, 0.0};
    }

    if (normalise_spins) {
      // ensure the spin is unit vector or a zero vector
      spin = unit_vector(spin);
    }

    for (auto n = 0; n < 3; ++n) {
        globals::s(i, n) = spin[n];
    }

    for (auto n = 0; n < 3; ++n) {
      globals::positions(i, n) = cartesian_positions_[i][n];
    }
  }

  bool initial_spin_state_is_a_file = lattice_settings.exists("spins");

  if (initial_spin_state_is_a_file) {
    std::string spin_filename = lattice_settings["spins"];

    cout << "  reading initial spin configuration from " << spin_filename << "\n";

    load_array_from_file(spin_filename, "/spins", globals::s);
  }
}

Vec3 Lattice::generate_cartesian_lattice_position_from_fractional(
    const Vec3 &unit_cell_frac_pos,
    const Vec3i &translation_vector) const
{
  return unitcell.matrix() * (unit_cell_frac_pos + translation_vector);
}

// generate a position within a periodic image of the entire system
Vec3 Lattice::generate_image_position(
        const Vec3 &unit_cell_cart_pos,
        const Vec3i &image_vector) const
{
  Vec3 frac_pos = cartesian_to_fractional(unit_cell_cart_pos);
  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      frac_pos[n] = frac_pos[n] + image_vector[n] * lattice_dimensions[n];
    }
  }
  return fractional_to_cartesian(frac_pos);
}

void Lattice::calc_symmetry_operations() {

  if (!symops_enabled_) {
    throw jams::runtime_error("Lattice::calc_symmetry_operations() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  cout << "  symmetry analysis\n";

  const char *wl = "abcdefghijklmnopqrstuvwxyz";

  double spg_lattice[3][3];
  // unit cell vectors have to be transposed because spglib wants
  // a set of 3 vectors rather than the unit cell matrix
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      spg_lattice[i][j] = unitcell.matrix()[i][j];
    }
  }

  double (*spg_positions)[3] = new double[motif_.size()][3];

  for (auto i = 0; i < motif_.size(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      spg_positions[i][j] = motif_[i].position[j];
    }
  }

  int *spg_types = new int[motif_.size()];

  for (auto i = 0; i < motif_.size(); ++i) {
    spg_types[i] = motif_[i].material_index;
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, motif_.size(), jams::defaults::lattice_tolerance);

  if (spglib_dataset_ == nullptr) {
    symops_enabled_ = false;
    jams_warning("spglib symmetry search failed, disabling symops");
    return;
  }

  cout << "    International " << spglib_dataset_->international_symbol << " (" <<  spglib_dataset_->spacegroup_number << ")\n";
  cout << "    Hall symbol " << spglib_dataset_->hall_symbol << "\n";
  cout << "    Hall number " << spglib_dataset_->hall_number << "\n";

  char ptsymbol[6];
  int pt_trans_mat[3][3];
  spg_get_pointgroup(ptsymbol,
           pt_trans_mat,
           spglib_dataset_->rotations,
           spglib_dataset_->n_operations);
  cout << "    point group  " << ptsymbol << "\n";
  cout << "    transformation matrix\n";
  for (auto i = 0; i < 3; i++ ) {
    cout << "    ";
    cout << spglib_dataset_->transformation_matrix[i][0] << " ";
    cout << spglib_dataset_->transformation_matrix[i][1] << " ";
    cout << spglib_dataset_->transformation_matrix[i][2] << "\n";
  }
  cout << "    Wyckoff letters ";
  for (auto i = 0; i < spglib_dataset_->n_atoms; i++ ) {
      cout << wl[spglib_dataset_->wyckoffs[i]] << " ";
  }
  cout << "\n";

  cout << "    equivalent atoms ";
  for (auto i = 0; i < spglib_dataset_->n_atoms; i++) {
    cout << spglib_dataset_->equivalent_atoms[i] << " ";
  }
  cout << "\n";

  if (verbose_is_enabled()) {
    cout << "    shifted lattice\n";
    cout << "    origin ";
    cout << spglib_dataset_->origin_shift[0] << " ";
    cout << spglib_dataset_->origin_shift[1] << " ";
    cout << spglib_dataset_->origin_shift[2] << "\n";

    cout << "    lattice vectors\n";
    for (auto i = 0; i < 3; ++i) {
      cout << "      ";
      for (auto j = 0; j < 3; ++j) {
        cout << spglib_dataset_->transformation_matrix[i][j] << " ";
      }
      cout << "\n";
    }

    cout << "    positions\n";
    for (int i = 0; i < motif_.size(); ++i) {
      double bij[3];
      matmul(spglib_dataset_->transformation_matrix, spg_positions[i], bij);
      cout << std::setw(12) << " ";
      cout << i << " ";
      cout << materials_.name(spg_types[i]) << " ";
      cout << bij[0] << " " << bij[1] << " " << bij[2] << "\n";
    }
  }

  cout << "    standard lattice\n";
  cout << "    std lattice vectors\n";

  for (int i = 0; i < 3; ++i) {
    cout << "    ";
    cout << spglib_dataset_->std_lattice[i][0] << " ";
    cout << spglib_dataset_->std_lattice[i][1] << " ";
    cout << spglib_dataset_->std_lattice[i][2] << "\n";
  }
  cout << "    num std atoms " << spglib_dataset_->n_std_atoms << "\n";

  cout << "    std_positions\n";
  for (int i = 0; i < spglib_dataset_->n_std_atoms; ++i) {
    cout << "    " << i << " " << materials_.name(spglib_dataset_->std_types[i]) << " ";
    cout << spglib_dataset_->std_positions[i][0] << " " << spglib_dataset_->std_positions[i][1] << " " << spglib_dataset_->std_positions[i][2] << "\n";
  }
  
  int primitive_num_atoms = motif_.size();
  double primitive_lattice[3][3];

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      primitive_lattice[i][j] = spg_lattice[i][j];
    }
  }

  double (*primitive_positions)[3] = new double[motif_.size()][3];

  for (auto i = 0; i < motif_.size(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      primitive_positions[i][j] = spg_positions[i][j];
    }
  }

  int *primitive_types = new int[motif_.size()];

  for (auto i = 0; i < motif_.size(); ++i) {
    primitive_types[i] = spg_types[i];
  }

  primitive_num_atoms = spg_find_primitive(primitive_lattice, primitive_positions, primitive_types, motif_.size(), jams::defaults::lattice_tolerance);

  // spg_find_primitive returns number of atoms in primitve cell
  if (primitive_num_atoms != motif_.size()) {
    cout << "\n";
    cout << "    !!! unit cell is not a primitive cell !!!\n";
    cout << "\n";
    cout << "    primitive lattice vectors:\n";

    for (int i = 0; i < 3; ++i) {
      cout << "    ";
      cout << jams::fmt::decimal << primitive_lattice[i][0] << " ";
      cout << jams::fmt::decimal << primitive_lattice[i][1] << " ";
      cout << jams::fmt::decimal << primitive_lattice[i][2] << "\n";
    }
    cout << "\n";
    cout << "    primitive motif positions:\n";

    int counter  = 0;
    for (int i = 0; i < primitive_num_atoms; ++i) {
      cout << "    " << counter << " " <<  materials_.name(primitive_types[i]) << " ";
      cout << primitive_positions[i][0] << " " << primitive_positions[i][1] << " " << primitive_positions[i][2] << "\n";
      counter++;
    }
  }
  cout << "\n";
  cout << "    num symops " << spglib_dataset_->n_operations << "\n";

  Mat3 rot;
  Mat3 id = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  for (auto n = 0; n < spglib_dataset_->n_operations; ++n) {

    if (verbose_is_enabled()) {
      cout << "    " << n << "\n---\n";
      for (auto i = 0; i < 3; ++i) {
        cout << "      ";
        for (auto j = 0; j < 3; ++j) {
          cout << spglib_dataset_->rotations[n][i][j] << " ";
        }
        cout << "\n";
      }
    }

    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        rot[i][j] = spglib_dataset_->rotations[n][i][j];
      }
    }

    rotations_.push_back(rot);
  }
  cout << "\n";

}


// reads an position in the fast integer space and applies the periodic boundaries
// if there are not periodic boundaries and this position is outside of the finite
// lattice then the function returns false
bool Lattice::apply_boundary_conditions(Vec3i& pos) const {
    for (int l = 0; l < 3; ++l) {
      if (!is_periodic(l) && (pos[l] < 0 || pos[l] >= globals::lattice->size(l))) {
        return false;
      } else {
        pos[l] = (pos[l] + globals::lattice->size(l)) % globals::lattice->size(l);
      }
    }
    return true;
}

bool Lattice::apply_boundary_conditions(int &a, int &b, int &c) const {
    if (!is_periodic(0) && (a < 0 || a >= globals::lattice->size(0))) {
      return false;
    } else {
      a = (a + globals::lattice->size(0)) % globals::lattice->size(0);
    }

    if (!is_periodic(1) && (b < 0 || b >= globals::lattice->size(1))) {
      return false;
    } else {
      b = (b + globals::lattice->size(1)) % globals::lattice->size(1);
    }

    if (!is_periodic(2) && (c < 0 || c >= globals::lattice->size(2))) {
      return false;
    } else {
      c = (c + globals::lattice->size(2)) % globals::lattice->size(2);
    }

    return true;
}

double Lattice::max_interaction_radius() const {
  return jams::maximum_interaction_length(supercell.a(), supercell.b(), supercell.c(), supercell.periodic());
}

// generate a vector of points which are symmetric to r_cart under the crystal symmetry
// the tolerance is used to determine if two points are equivalent
std::vector<Vec3> Lattice::generate_symmetric_points(const Vec3 &r_cart, const double &tolerance = jams::defaults::lattice_tolerance) const {

  const auto r_frac = cartesian_to_fractional(r_cart);
  std::vector<Vec3> symmetric_points;

  // store the original point
  symmetric_points.push_back(r_cart);
  // loop through all of the symmmetry operations
  for (const auto rotation_matrix : rotations_) {
    // apply a symmetry operation
    const auto r_sym = fractional_to_cartesian(rotation_matrix * r_frac);

    // check if the generated point is already in the vector
    if (!vec_exists_in_container(symmetric_points, r_sym, tolerance)) {
      // it's not in the vector so append it
      symmetric_points.push_back(r_sym);
    }
  }

  return symmetric_points;
}

bool Lattice::is_a_symmetry_complete_set(const std::vector<Vec3> &points, const double &tolerance = jams::defaults::lattice_tolerance) const {
  // loop over the collection of points
  for (const auto r : points) {
    // for each point generate the symmetric points according to the the crystal symmetry
    for (const auto r_sym : generate_symmetric_points(r, tolerance)) {
      // if a symmetry generated point is not in our original collection of points then our original collection was not a complete set
      // and we return fals
      if (!vec_exists_in_container(points, r_sym, tolerance)) {
        return false;
      }
    }
  }
  // the collection of points contains all allowed symmetric points
  return true;
}

const Atom &Lattice::motif_atom(const int &i) const {
  return motif_[i];
}

const Material &Lattice::material(const int &i) const {
  return materials_[i];
}

const Cell &Lattice::get_supercell() {
  return supercell;
}

const Cell &Lattice::get_unitcell() {
  return unitcell;
}

const Mat3 &Lattice::get_global_rotation_matrix() {
  return global_orientation_matrix_;
}

bool Lattice::material_exists(const std::string &name) const {
  return materials_.contains(name);
}

Lattice::ImpurityMap Lattice::read_impurities_from_config(const libconfig::Setting &settings) {
  Lattice::ImpurityMap impurities;

  size_t materialA, materialB;
  for (auto n = 0; n < settings.getLength(); ++n) {
    try {
      materialA = materials_.id(settings[n][0].c_str());
    }
    catch(std::out_of_range &e) {
      jams_die("impurity %d materialA (%s) does not exist", n, settings[n][0].c_str());
    }

    try {
      materialB = materials_.id(settings[n][1].c_str());
    }
    catch(std::out_of_range &e) {
      jams_die("impurity %d materialB (%s) does not exist", n, settings[n][1].c_str());
    }

    auto fraction  = double(settings[n][2]);

    if (fraction < 0.0 || fraction >= 1.0) {
      jams_die("impurity %d fraction must be 0 =< x < 1", n);
    }

    Impurity imp = {materialB, fraction};

    if(impurities.emplace(materialA, imp).second == false) {
      jams_die("impurity %d defined redefines a previous impurity", n);
    }
  }
  return impurities;
}

unsigned Lattice::atom_motif_position(const int &i) const {
  return atoms_[i].motif_index;
}

bool Lattice::has_impurities() const {
    return !impurity_map_.empty();
}

const Vec3 &Lattice::atom_fractional_position(const int &i) const {
  return fractional_positions_[i];
}

const std::vector<Vec3> &Lattice::atom_cartesian_positions() const {
  return cartesian_positions_;
}

double jams::maximum_interaction_length(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b& periodic_boundaries) {
  // 3D periodic
  // -----------
  // This must be the inradius of the parellelepiped
  if (periodic_boundaries == Vec3b{true, true, true}) {
    return jams::maths::parallelepiped_inradius(a, b, c);
  }

  // 2D periodic
  // -----------
  // We only care about the interaction radius in a single plane. In the
  // 'open' direction it doesn't matter what the interaction length is because
  // there will not be any atoms to interact with beyond the boundary.
  // Which plane to use is defined by which of the two dimensions are periodic.
  if (periodic_boundaries == Vec3b{true, true, false}) {
    return jams::maths::parallelogram_inradius(a, b);
  }
  if (periodic_boundaries == Vec3b{true, false, true}) {
    return jams::maths::parallelogram_inradius(a, c);
  }
  if (periodic_boundaries == Vec3b{false, true, true}) {
    return jams::maths::parallelogram_inradius(b, c);
  }

  // 1D periodic
  // -----------
  // As with 2D, we only care about the interaction along 1 dimension for the
  // purposes of self interaction. Here is simply half the length along that
  // dimension.
  if (periodic_boundaries == Vec3b{true, false, false}) {
    return 0.5 * norm(a);
  }
  if (periodic_boundaries == Vec3b{false, true, false}) {
    return 0.5 * norm(b);
  }
  if (periodic_boundaries == Vec3b{false, false, true}) {
    return 0.5 * norm(c);
  }

  // Open system (not periodic)
  // --------------------------
  // In an open system we can have any interaction radius because there is no
  // possibility for self interaction. But we should return some meaningful
  // number here (not inf!). The largest possible interaction length for a
  // parallelepiped is the longest of the body diagonals.
  return jams::maths::parallelepiped_longest_diagonal(a, b, c);
}