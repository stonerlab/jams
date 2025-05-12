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
    /// Return fractional coordinate normalised to the range [0,1)
    Vec3 normalise_fractional_coordinate(Vec3 r_frac, const double eps = jams::defaults::lattice_tolerance) {
      for (auto n = 0; n < 3; ++n) {
        if (r_frac[n] < 0.0) {
          r_frac[n] = r_frac[n] + 1.0;
        }
        // If we end up exactly on the opposite face/edge of the cell then
        // this should actually be in the next cell (i.e. fractional coordinates
        // are in the range 0 <= r_frac[n] < 1. So we must map coordinates equal to 1
        // back to 0.
        if (approximately_equal(r_frac[n], 1.0, eps)) {
          r_frac[n] = 0.0;
        }
      }
      return r_frac;
    }

    /// Returns true if the fractional coordinate is correctly normalised in the range [0, 1)
    bool is_fractional_coordinate_normalised(const Vec3 &r_frac, const double eps = jams::defaults::lattice_tolerance) {
      // check fractional coordinates are in the range 0 <= r_frac[n] < 1
      for (auto n = 0; n < 3; ++n) {
        if (r_frac[n] < 0.0 || r_frac[n] > 1.0 || approximately_equal(r_frac[n], 1.0, eps)) {
          return false;
        }
      }
      return true;
    }

    void output_basis_vectors(const Cell& cell) {
      cout << "    a1 = " << jams::fmt::decimal << cell.a1() << "\n";
      cout << "    a2 = " << jams::fmt::decimal << cell.a2() << "\n";
      cout << "    a3 = " << jams::fmt::decimal << cell.a3() << "\n";
    }

    void output_reciprocal_basis_vectors(const Cell& cell) {
      cout << "    b1 = " << jams::fmt::decimal << cell.b1() << "\n";
      cout << "    b2 = " << jams::fmt::decimal << cell.b2() << "\n";
      cout << "    b3 = " << jams::fmt::decimal << cell.b3() << "\n";
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


int Lattice::size(int dimension) const {
  return lattice_dimensions_[dimension];
}

Vec3i Lattice::size() const {
  return lattice_dimensions_;
}

int Lattice::num_basis_sites() const {
  return basis_sites_.size();
}

Vec3 Lattice::a1() const {
  return unitcell.a1();
}

Vec3 Lattice::a2() const {
  return unitcell.a2();
}

Vec3 Lattice::a3() const {
  return unitcell.a3();
}

int
Lattice::num_materials() const {
  return materials_.size();
}

std::string
Lattice::material_name(int material_index) const {
  return materials_.name(material_index);
}

int
Lattice::material_index(const std::string &material_name) const {
  return materials_.id(material_name);
}

int
Lattice::lattice_site_material_id(int lattice_site_index) const {
  assert(lattice_site_index < lattice_sites_.size());
  return lattice_sites_[lattice_site_index].material_index;
}

std::string
Lattice::lattice_site_material_name(int lattice_site_index) const {
  assert(lattice_site_index < lattice_sites_.size());
  return material_name(lattice_site_material_id(lattice_site_index));
}

const Vec3 &
Lattice::lattice_site_position_cart(int lattice_site_index) const {
  return lattice_site_positions_cart_[lattice_site_index];
}

Vec3
Lattice::displacement(const Vec3 &position_i_cart, const Vec3 &position_j_cart) const {
  return jams::minimum_image(supercell.a1(),
                             supercell.a2(),
                             supercell.a3(), supercell.periodic(), position_i_cart, position_j_cart, jams::defaults::lattice_tolerance);
}

Vec3 Lattice::displacement(const unsigned &lattice_site_i, const unsigned &lattice_site_j) const {
  return jams::minimum_image(supercell.a1(),
                             supercell.a2(),
                             supercell.a3(), supercell.periodic(), lattice_sites_[lattice_site_i].position_frac, lattice_sites_[lattice_site_j].position_frac, jams::defaults::lattice_tolerance);
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
  assert(i < lattice_dimensions_[0]);
  assert(i >= 0);
  assert(j < lattice_dimensions_[1]);
  assert(j >= 0);
  assert(k < lattice_dimensions_[2]);
  assert(k >= 0);
  assert(m < num_basis_sites());
  assert(m >= 0);

  return lattice_map_(i, j, k, m);
}

bool Lattice::is_periodic(int dimension) const {
  return lattice_periodic[dimension];
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
    jams::interaction_calculator(unitcell, basis_sites_, interaction_calculator_radius);
  }

  if (symops_enabled_) {

    if (basis_sites_.size() > jams::defaults::warning_unitcell_symops_size) {
      jams_warning("symmetry calculation may be slow as unit cell has more than %d atoms and symops is turned on", jams::defaults::warning_unitcell_symops_size);
    }

    calc_symmetry_operations();
  }

  generate_supercell(cfg.lookup("lattice"));
}

void Lattice::read_basis_sites_from_config(const libconfig::Setting &positions, CoordinateFormat coordinate_format) {
  Atom atom;
  std::string atom_name;

  basis_sites_.clear();

  for (int i = 0; i < positions.getLength(); ++i) {
    atom_name = positions[i][0].c_str();

    // check the material type is defined
    if (!materials_.contains(atom_name)) {
      throw jams::runtime_error("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material_index = materials_.id(atom_name);

    atom.position_frac[0] = positions[i][1][0];
    atom.position_frac[1] = positions[i][1][1];
    atom.position_frac[2] = positions[i][1][2];

    if (coordinate_format == CoordinateFormat::CARTESIAN) {
      atom.position_frac = cartesian_to_fractional(atom.position_frac);
    }

    atom.position_frac = normalise_fractional_coordinate(atom.position_frac);

    if (!is_fractional_coordinate_normalised(atom.position_frac)) {
      throw std::runtime_error("atom position " + std::to_string(i) + " is not a valid fractional coordinate");
    }

    atom.id = basis_sites_.size();

    basis_sites_.push_back(atom);
  }
}

void Lattice::read_basis_sites_from_file(const std::string &filename, CoordinateFormat coordinate_format) {
  std::string line;
  std::ifstream position_file(filename.c_str());

  if(position_file.fail()) {
    throw jams::runtime_error("failed to open position file " + filename, __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  basis_sites_.clear();

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
    line_as_stream >> atom_name >> atom.position_frac[0] >> atom.position_frac[1] >> atom.position_frac[2];

    if (coordinate_format == CoordinateFormat::CARTESIAN) {
      atom.position_frac = cartesian_to_fractional(atom.position_frac);
    }

    atom.position_frac = normalise_fractional_coordinate(atom.position_frac);

    if (!is_fractional_coordinate_normalised(atom.position_frac)) {
      throw std::runtime_error("atom position " + std::to_string(basis_sites_.size()) + " is not a valid fractional coordinate");
    }
    // check the material type is defined
    if (!materials_.contains(atom_name)) {
      throw jams::runtime_error("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material_index = materials_.id(atom_name);
    atom.id = basis_sites_.size();

    basis_sites_.push_back(atom);
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
  output_basis_vectors(unitcell);
  cout << "\n";

  cout << "    unit cell (matrix form)\n";

  for (auto i = 0; i < 3; ++i) {
    cout << "    " << jams::fmt::decimal << unitcell.matrix()[i] << "\n";
  }
  cout << "\n";

  cout << "    unit cell inverse vectors\n";
  output_reciprocal_basis_vectors(unitcell);
  cout << "\n";

  cout << "    inverse unit cell (matrix form)\n";
  for (auto i = 0; i < 3; ++i) {
    cout << "    " << jams::fmt::decimal << unitcell.inverse_matrix()[i] << "\n";
  }
  cout << "\n";
}

void Lattice::read_lattice_from_config(const libconfig::Setting &settings) {
  lattice_periodic = jams::config_optional<Vec3b>(settings, "periodic", jams::defaults::lattice_periodic_boundaries);
  lattice_dimensions_ = jams::config_required<Vec3i>(settings, "size");

  supercell = scale(Cell(unitcell.matrix(), lattice_periodic), lattice_dimensions_);

  cout << "  lattice\n";
  cout << "    size " << lattice_dimensions_ << " (unit cells)\n";
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
      throw jams::ConfigException(lattice_settings, "Only one of 'orientation_lattice_vector' or 'orientation_cartesian_vector' can be defined");
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
    read_basis_sites_from_config(unitcell_settings["positions"], cfg_coordinate_format);
  } else {
    position_filename = unitcell_settings["positions"].c_str();
    read_basis_sites_from_file(position_filename, cfg_coordinate_format);
  }

  cout << "  motif positions " << position_filename << "\n";
  cout << "  format " << cfg_coordinate_format_name << "\n";

  for (const Atom &atom: basis_sites_) {
    cout << "    " << jams::fmt::integer << atom.id << " ";
    cout << materials_.name(atom.material_index) << " ";
    cout << jams::fmt::decimal << atom.position_frac << "\n";
  }
  cout << endl;

  bool check_closeness = jams::config_optional<bool>(unitcell_settings, "check_closeness", true);

  if (check_closeness) {
    cout << "checking no atoms are too close together..." << std::flush;

    for (auto i = 0; i < basis_sites_.size(); ++i) {
      for (auto j = i + 1; j < basis_sites_.size(); ++j) {
        auto distance = norm(
            jams::minimum_image(unitcell.a1(), unitcell.a2(), unitcell.a3(),
                                unitcell.periodic(),
                                fractional_to_cartesian(basis_sites_[i].position_frac),
                                fractional_to_cartesian(basis_sites_[j].position_frac),
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
    throw jams::SanityException("unitcell volume has changed after rotation");
  }

  cout << "  global rotated lattice vectors\n";
  output_basis_vectors(unitcell);
  cout << "\n";
  cout << "  global rotated inverse vectors\n";
  output_reciprocal_basis_vectors(unitcell);
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
    throw jams::SanityException("unitcell volume has changed after rotation");
  }

  cout << "  oriented lattice vectors\n";
  output_basis_vectors(unitcell);
  cout << "\n";

  cout << "  oriented inverse vectors\n";
  output_reciprocal_basis_vectors(unitcell);
  cout << "\n";
}


void Lattice::generate_supercell(const libconfig::Setting &lattice_settings)
{
  Vec3i kmesh_size = {lattice_dimensions_[0], lattice_dimensions_[1], lattice_dimensions_[2]};

  if (!lattice_periodic[0] || !lattice_periodic[1] || !lattice_periodic[2]) {
    cout << "\nzero padding non-periodic dimensions\n";
    // double any non-periodic dimensions for zero padding
    for (auto i = 0; i < 3; ++i) {
      if (!lattice_periodic[i]) {
        kmesh_size[i] = 2*lattice_dimensions_[i];
      }
    }
    cout << "\npadded kspace size " << kmesh_size << "\n";
  }

  kspace_size_ = {kmesh_size[0], kmesh_size[1], kmesh_size[2]};
  kspace_map_.resize(kspace_size_[0], kspace_size_[1], kspace_size_[2]);
  kspace_map_.fill(-1);



  cout << "\nkspace size " << kmesh_size << "\n";



  lattice_map_.resize(this->size(0), this->size(1), this->size(2), this->num_basis_sites());
  // initialize everything to -1 so we can check for double assignment below
  lattice_map_.fill(-1);

  const auto num_cells = product(lattice_dimensions_);
  const auto expected_num_atoms = num_basis_sites() * num_cells;

  cell_centers_.reserve(num_cells);
  cell_offsets_.reserve(num_cells);
  lattice_sites_.reserve(expected_num_atoms);
  lattice_site_to_cell_lookup_.reserve(expected_num_atoms);

  auto impurity_rand = std::bind(std::uniform_real_distribution<>(), pcg32(impurity_seed_));

  // loop over the translation vectors for lattice size
  int atom_counter = 0;
  std::vector<size_t> type_counter(materials_.size(), 0);


  unsigned cell_counter = 0;
  for (auto i = 0; i < lattice_dimensions_[0]; ++i) {
    for (auto j = 0; j < lattice_dimensions_[1]; ++j) {
      for (auto k = 0; k < lattice_dimensions_[2]; ++k) {
        auto cell_offset = Vec3i{{i, j, k}};
        cell_offsets_.push_back(cell_offset);
        cell_centers_.push_back(generate_cartesian_lattice_position_from_fractional(Vec3{0.5,0.5,0.5}, cell_offset));

        for (auto m = 0; m < basis_sites_.size(); ++m) {
          auto position    = generate_cartesian_lattice_position_from_fractional(basis_sites_[m].position_frac, cell_offset);
          auto material    = basis_sites_[m].material_index;

          if (impurity_map_.count(material)) {
            auto impurity    = impurity_map_[material];

            if (impurity_rand() < impurity.fraction) {
              material = impurity.material;
            }
          }

          lattice_sites_.push_back({atom_counter, material, m, position});

          lattice_site_positions_cart_.push_back(position);
          lattice_site_positions_frac_.push_back(cartesian_to_fractional(position));

          lattice_site_to_cell_lookup_.push_back(cell_counter);

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
    throw jams::SanityException("the number of computed lattice sites was zero, check input");
  }

  cout << "    lattice material count\n";
  for (auto n = 0; n < type_counter.size(); ++n) {
    cout << "      " << materials_.name(n) << ": " << type_counter[n] << "\n";
  }

  // this is the top right hand corner of the top right unit cell in the super cell
  rmax_ = generate_cartesian_lattice_position_from_fractional(Vec3{0.0, 0.0, 0.0}, lattice_dimensions_);

  globals::num_spins = atom_counter;
  globals::num_spins3 = 3*atom_counter;

  cout << "  computed lattice positions " << atom_counter << "\n";
  for (auto i = 0; i < lattice_sites_.size(); ++i) {
    cout << "    " << jams::fmt::fixed_integer << i << " ";
    cout << std::setw(8) << materials_.name(lattice_sites_[i].material_index) << " ";
    cout << jams::fmt::decimal << lattice_sites_[i].position_frac << " ";
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
    const auto material = materials_[lattice_site_material_id(i)];

    globals::mus(i)   = material.moment;
    globals::alpha(i) = material.alpha;

    if (use_gilbert_prefactor) {
      globals::gyro(i)  = jams::gilbert_gyro_prefactor(material.gyro, material.alpha, material.moment);
    } else {
      globals::gyro(i) = jams::landau_lifshitz_gyro_prefactor(material.gyro, material.alpha, material.moment);
    }

    Vec3 spin = material.spin;

    if (material.randomize) {
      spin = uniform_random_sphere(rng);
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
      globals::positions(i, n) = lattice_site_positions_cart_[i][n];
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
    const Vec3 &basis_site_position_frac,
    const Vec3i &lattice_translation_vector) const
{
  return unitcell.matrix() * (basis_site_position_frac + lattice_translation_vector);
}

// generate a position within a periodic image of the entire system
Vec3 Lattice::generate_image_position(
        const Vec3 &unit_cell_cart_pos,
        const Vec3i &image_vector) const
{
  Vec3 frac_pos = cartesian_to_fractional(unit_cell_cart_pos);
  for (int n = 0; n < 3; ++n) {
    if (is_periodic(n)) {
      frac_pos[n] = frac_pos[n] + image_vector[n] * lattice_dimensions_[n];
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

  double (*spg_positions)[3] = new double[basis_sites_.size()][3];

  for (auto i = 0; i < basis_sites_.size(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      spg_positions[i][j] = basis_sites_[i].position_frac[j];
    }
  }

  int *spg_types = new int[basis_sites_.size()];

  for (auto i = 0; i < basis_sites_.size(); ++i) {
    spg_types[i] = basis_sites_[i].material_index;
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, basis_sites_.size(), jams::defaults::lattice_tolerance);

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
    for (int i = 0; i < basis_sites_.size(); ++i) {
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

  int primitive_num_atoms = basis_sites_.size();
  double primitive_lattice[3][3];

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      primitive_lattice[i][j] = spg_lattice[i][j];
    }
  }

  double (*primitive_positions)[3] = new double[basis_sites_.size()][3];

  for (auto i = 0; i < basis_sites_.size(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      primitive_positions[i][j] = spg_positions[i][j];
    }
  }

  int *primitive_types = new int[basis_sites_.size()];

  for (auto i = 0; i < basis_sites_.size(); ++i) {
    primitive_types[i] = spg_types[i];
  }

  primitive_num_atoms = spg_find_primitive(primitive_lattice, primitive_positions, primitive_types, basis_sites_.size(), jams::defaults::lattice_tolerance);

  // spg_find_primitive returns number of atoms in primitve cell
  if (primitive_num_atoms != basis_sites_.size()) {
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
  Vec3 trans;
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

    for (auto i = 0; i < 3; ++i) {
      trans[i] = spglib_dataset_->translations[n][i];
    }

    sym_rotations_.push_back(rot);
    sym_translations_.push_back(trans);
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
  return jams::maximum_interaction_length(supercell.a1(), supercell.a2(), supercell.a3(), supercell.periodic());
}

// generate a vector of points which are symmetric to r_cart under the crystal symmetry
// the tolerance is used to determine if two points are equivalent
std::vector<Vec3> Lattice::generate_symmetric_points(int basis_site_index, const Vec3 &r_cart, const double &tolerance = jams::defaults::lattice_tolerance) {

  const auto r_frac = cartesian_to_fractional(r_cart);
  std::vector<Vec3> symmetric_points;

  // store the original point
  symmetric_points.push_back(r_cart);
  // loop through all of the symmmetry operations
  for (const auto rotation_matrix : lattice_site_point_group_symops(basis_site_index)) {
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

bool Lattice::is_a_symmetry_complete_set(const int motif_index, const std::vector<Vec3> &points, const double &tolerance = jams::defaults::lattice_tolerance) {
  // loop over the collection of points
  for (const auto r : points) {
    // for each point generate the symmetric points according to the the crystal symmetry
    for (const auto r_sym : generate_symmetric_points(motif_index, r, tolerance)) {
      // if a symmetry generated point is not in our original collection of points then our original collection was not a complete set
      // and we return false
      if (!vec_exists_in_container(points, r_sym, tolerance)) {
        return false;
      }
    }
  }
  // the collection of points contains all allowed symmetric points
  return true;
}

const Atom &Lattice::basis_site_atom(const int &basis_site_index) const {
  return basis_sites_[basis_site_index];
}

const Material &Lattice::material(const int &material_index) const {
  return materials_[material_index];
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

bool Lattice::material_exists(const std::string &material_name) const {
  return materials_.contains(material_name);
}

Lattice::ImpurityMap Lattice::read_impurities_from_config(const libconfig::Setting &settings) {
  Lattice::ImpurityMap impurities;

  size_t materialA, materialB;
  for (auto n = 0; n < settings.getLength(); ++n) {
    try {
      materialA = materials_.id(settings[n][0].c_str());
    }
    catch(std::out_of_range &e) {
      jams::ConfigException(settings, "impurity ", n, " materialA (", settings[n][0].c_str(), ") does not exist");
    }

    try {
      materialB = materials_.id(settings[n][1].c_str());
    }
    catch(std::out_of_range &e) {
      jams::ConfigException(settings, "impurity ", n, " materialB (", settings[n][1].c_str(), ") does not exist");
    }

    auto fraction  = double(settings[n][2]);

    if (fraction < 0.0 || fraction >= 1.0) {
      jams::ConfigException(settings, "impurity ", n, " fraction must be 0 =< x < 1");
    }

    Impurity imp = {materialB, fraction};

    if(impurities.emplace(materialA, imp).second == false) {
      jams::ConfigException(settings, "impurity ", n, " redefines a previous impurity");
    }
  }
  return impurities;
}

unsigned Lattice::lattice_site_basis_index(int lattice_site_index) const {
  return lattice_sites_[lattice_site_index].basis_site_index;
}

bool Lattice::has_impurities() const {
    return !impurity_map_.empty();
}

const Vec3 &Lattice::lattice_site_vector_frac(int lattice_site_index) const {
  return lattice_site_positions_frac_[lattice_site_index];
}

const std::vector<Vec3> &Lattice::lattice_site_positions_cart() const {
  return lattice_site_positions_cart_;
}

const std::vector<Mat3> &Lattice::lattice_site_point_group_symops(int lattice_site_index) {
    assert(lattice_site_index >= 0);
    assert(lattice_site_index < num_basis_sites());
    // Pre-calculate the symops the first time the function is called
    if (basis_site_point_group_symops_.empty()) {
        basis_site_point_group_symops_.resize(num_basis_sites());
        for (auto m = 0; m < num_basis_sites(); ++m) {
            auto motif_position = basis_site_atom(m).position_frac;
            for (auto n = 0; n < sym_translations_.size(); ++n) {
                // The point groups are found only from space group operations which do not include translation
                // so we must skip any elements with a translation.
                if (!approximately_zero(sym_translations_[n], jams::defaults::lattice_tolerance)) {
                  continue;
                }

                auto rotation = sym_rotations_[n];

                auto new_position = normalise_fractional_coordinate(rotation * motif_position);
                // TODO: need to translate back into unit cell
                if  (approximately_equal(motif_position, new_position, jams::defaults::lattice_tolerance)) {
                    basis_site_point_group_symops_[m].push_back(rotation);
                }
            }
        }
    }
    return basis_site_point_group_symops_[lattice_site_index];
}

double jams::maximum_interaction_length(const Vec3 &a1, const Vec3 &a2, const Vec3 &a3, const Vec3b& periodic_boundaries) {
  // 3D periodic
  // -----------
  // This must be the inradius of the parellelepiped
  if (periodic_boundaries == Vec3b{true, true, true}) {
    return jams::maths::parallelepiped_inradius(a1, a2, a3);
  }

  // 2D periodic
  // -----------
  // We only care about the interaction radius in a1 single plane. In the
  // 'open' direction it doesn't matter what the interaction length is because
  // there will not be any atoms to interact with beyond the boundary.
  // Which plane to use is defined by which of the two dimensions are periodic.
  if (periodic_boundaries == Vec3b{true, true, false}) {
    return jams::maths::parallelogram_inradius(a1, a2);
  }
  if (periodic_boundaries == Vec3b{true, false, true}) {
    return jams::maths::parallelogram_inradius(a1, a3);
  }
  if (periodic_boundaries == Vec3b{false, true, true}) {
    return jams::maths::parallelogram_inradius(a2, a3);
  }

  // 1D periodic
  // -----------
  // As with 2D, we only care about the interaction along 1 dimension for the
  // purposes of self interaction. Here is simply half the length along that
  // dimension.
  if (periodic_boundaries == Vec3b{true, false, false}) {
    return 0.5 * norm(a1);
  }
  if (periodic_boundaries == Vec3b{false, true, false}) {
    return 0.5 * norm(a2);
  }
  if (periodic_boundaries == Vec3b{false, false, true}) {
    return 0.5 * norm(a3);
  }

  // Open system (not periodic)
  // --------------------------
  // In an open system we can have any interaction radius because there is no
  // possibility for self interaction. But we should return some meaningful
  // number here (not inf!). The largest possible interaction length for a1
  // parallelepiped is the longest of the body diagonals.
  return jams::maths::parallelepiped_longest_diagonal(a1, a2, a3);
}
