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
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include <cfloat>
#include <pcg/pcg_random.hpp>

#include "H5Cpp.h"

#include "jams/helpers/defaults.h"
#include "jams/containers/material.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
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
      cout << "    a = " << cell.a() << "\n";
      cout << "    b = " << cell.b() << "\n";
      cout << "    c = " << cell.c() << "\n";
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


double Lattice::parameter() const {
  return lattice_parameter;
}

double Lattice::volume() const {
  return ::volume(supercell) * pow3(lattice_parameter);
}

int Lattice::size(int i) const {
  return lattice_dimensions[i];
}

int Lattice::num_motif_positions() const {
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

Vec3
Lattice::motif_position_frac(int i) const {
  assert(i < num_motif_positions());
  return motif_[i].pos;
}

Vec3
Lattice::motif_position_cart(int i) const {
  assert(i < num_motif_positions());
  return unitcell.matrix() * motif_[i].pos;
}

Material
Lattice::motif_material(int i) const {
  assert(i < motif_.size());
  return materials_[motif_[i].material];
}

int
Lattice::num_materials() const {
  return materials_.size();
}

std::string
Lattice::material_name(int uid) {
  return materials_.name(uid);
}

int
Lattice::material_id(const string &name) {
  return materials_.id(name);
}

int
Lattice::atom_material_id(const int &i) const {
  assert(i < atoms_.size());
  return atoms_[i].material;
}

Vec3
Lattice::atom_position(const int &i) const {
  return atoms_[i].pos;
}

void
Lattice::atom_neighbours(const int &i, const double &r_cutoff, std::vector<Atom> &neighbours) const {
  neartree_->find_in_radius(r_cutoff, neighbours, {i, atoms_[i].material, atoms_[i].pos});
}

Vec3
Lattice::displacement(const Vec3 &r_i, const Vec3 &r_j) const {
  return minimum_image(supercell, r_i, r_j);
}

Vec3
Lattice::cartesian_to_fractional(const Vec3 &r_cart) const {
  return unitcell.inverse_matrix() * r_cart;
}

Vec3
Lattice::fractional_to_cartesian(const Vec3 &r_frac) const {
  return unitcell.matrix() * r_frac;
}

Vec3
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
  assert(m < num_motif_positions());
  assert(m >= 0);

  return lattice_map_(i, j, k, m);
}

bool Lattice::is_periodic(int i) const {
  return lattice_periodic[i];
}

const Vec3i &Lattice::supercell_index(const int &i) const {
  return supercell_indicies_[i];
}

const Vec3i &Lattice::kspace_size() const {
  return kspace_size_;
}

void Lattice::init_from_config(const libconfig::Config& cfg) {

  set_name("lattice");
  set_verbose(jams::config_optional<bool>(cfg.lookup("lattice"), "verbose", false));
  set_debug(jams::config_optional<bool>(cfg.lookup("lattice"), "debug", false));

  symops_enabled_ = jams::config_optional<bool>(cfg.lookup("lattice"), "symops", jams::default_lattice_symops);

  cout << "  symops " << symops_enabled_ << "\n";

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
    if (!materials_.contains(atom_name)) {
      throw general_exception("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material = materials_.id(atom_name);

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
    if (!materials_.contains(atom_name)) {
      throw general_exception("material " + atom_name + " in the motif is not defined in the configuration", __FILE__, __LINE__, __PRETTY_FUNCTION__);
    }
    atom.material = materials_.id(atom_name);
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

    cout << "    " << material.id << " " << material.name << "\n";
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
    throw general_exception("lattice parameter cannot be negative", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

  if (lattice_parameter > 1e-7) {
    jams_warning("lattice parameter is unusually large - units should be meters");
  }

  cout << "  unit cell\n";
  cout << "    parameter " << lattice_parameter << "\n";
  cout << "    volume " << this->volume() << "\n";
  cout << "\n";

  cout << "    unit cell vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";

  cout << "    unit cell (matrix form)\n";

  for (auto i = 0; i < 3; ++i) {
    cout << "    " << unitcell.matrix()[i] << "\n";
  }
  cout << "\n";

  cout << "    inverse unit cell (matrix form)\n";
  for (auto i = 0; i < 3; ++i) {
    cout << "    " << unitcell.inverse_matrix()[i] << "\n";
  }
  cout << "\n";
}

void Lattice::read_lattice_from_config(const libconfig::Setting &settings) {
  lattice_periodic = jams::config_optional<Vec3b>(settings, "periodic", jams::default_lattice_periodic_boundaries);
  lattice_dimensions = jams::config_required<Vec3i>(settings, "size");

  cout << "  lattice\n";
  cout << "    size " << lattice_dimensions << "\n";
  cout << "    periodic " << lattice_periodic << "\n";
  cout << "\n";
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

  cout << "  motif positions " << position_filename << "\n";
  cout << "  format " << cfg_coordinate_format_name << "\n";

  for (const Atom &atom: motif_) {
    cout << "    " << atom.id << " " <<  materials_.name(atom.material) << " " << atom.pos << "\n";
  }
  cout << "\n";

}

void Lattice::global_rotation(const Mat3& rotation_matrix) {
  auto volume_before = ::volume(unitcell);

  unitcell = rotate(unitcell, rotation_matrix);
  supercell = rotate(supercell, rotation_matrix);

  auto volume_after = ::volume(unitcell);

  if (std::abs(volume_before - volume_after) > 1e-6) {
    jams_error("unitcell volume has changed after rotation");
  }

  cout << "  global rotated lattice vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";
}

void Lattice::global_reorientation(const Vec3 &reference, const Vec3 &vector) {

  Vec3 orientation_cartesian_vector = normalize(unitcell.matrix() * vector);

  cout << "  orientation_axis " << reference << "\n";
  cout << "  orientation_lattice_vector " << vector << "\n";
  cout << "  orientation_cartesian_vector " << orientation_cartesian_vector << "\n";

  Mat3 orientation_matrix = rotation_matrix_between_vectors(orientation_cartesian_vector, reference);

  cout << "  orientation rotation matrix \n";
  cout << "    " << orientation_matrix[0] << "\n";
  cout << "    " << orientation_matrix[1] << "\n";
  cout << "    " << orientation_matrix[2] << "\n";
  cout << "\n";

  Vec3 rotated_orientation_vector = orientation_matrix * orientation_cartesian_vector;

  if (verbose_is_enabled()) {
    cout << "  rotated_orientation_vector\n";
    cout << "    " << rotated_orientation_vector << "\n";
  }

  auto volume_before = ::volume(unitcell);
  rotate(unitcell, orientation_matrix);
  rotate(supercell, orientation_matrix);
  auto volume_after = ::volume(unitcell);

  if (std::abs(volume_before - volume_after) > 1e-6) {
    jams_error("unitcell volume has changed after rotation");
  }

  cout << "  oriented lattice vectors\n";
  output_unitcell_vectors(unitcell);
  cout << "\n";
}


void Lattice::init_lattice_positions(const libconfig::Setting &lattice_settings)
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


  for (auto i = 0; i < product(kspace_size_); ++i) {
    kspace_map_[i] = -1;
  }


  const auto expected_num_atoms = num_motif_positions() * product(lattice_dimensions);

  lattice_map_.resize(this->size(0), this->size(1), this->size(2), this->num_motif_positions());
  for (auto i = 0; i < expected_num_atoms; ++i) {
    // initialize everything to -1 so we can check for double assignment below
    lattice_map_[i] = -1;
  }

  supercell_indicies_.reserve(expected_num_atoms);
  atoms_.reserve(expected_num_atoms);

  // loop over the translation vectors for lattice size
  int atom_counter = 0;
  for (auto i = 0; i < lattice_dimensions[0]; ++i) {
    for (auto j = 0; j < lattice_dimensions[1]; ++j) {
      for (auto k = 0; k < lattice_dimensions[2]; ++k) {
        for (auto m = 0; m < motif_.size(); ++m) {

          auto translation = Vec3i{{i, j, k}};
          auto position    = generate_position(motif_[m].pos, translation);
          auto material    = motif_[m].material;

          atoms_.push_back({atom_counter, material, position});
          supercell_indicies_.push_back(translation);

          // number the site in the fast integer lattice
          lattice_map_(i, j, k, m) = atom_counter;

          atom_counter++;
        }
      }
    }
  }

  // store max coordinates
  rmax_[0] = -DBL_MAX; rmax_[1] = -DBL_MAX; rmax_[2] = -DBL_MAX;
  for (const auto& a : atoms_) {
    for (auto n = 0; n < 3; ++n) {
      if (a.pos[n] > rmax_[n]) {
        rmax_[n] = a.pos[n];
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

  cout << "  computed lattice positions " << atom_counter << "\n";
  for (auto i = 0; i < atoms_.size(); ++i) {
    cout << i << " " << materials_.name(atoms_[i].material)  << " " << atoms_[i].pos << " " << supercell_index(i) << "\n";
    if(!verbose_is_enabled() && i > 7) {
      cout << "    ... [use verbose output for details] ... \n";
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

  pcg32 rng = pcg_extras::seed_seq_from<std::random_device>();
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto material = materials_[atom_material_id(i)];

    globals::mus(i)   = material.moment;
    globals::alpha(i) = material.alpha;
    globals::gyro(i)  = jams::llg_gyro_prefactor(material.gyro, material.alpha, material.moment);

    for (auto n = 0; n < 3; ++n) {
      globals::s(i, n) = material.spin[n];
    }

    if (material.randomize) {
      Vec3 s_init = uniform_random_sphere(rng);
      for (auto n = 0; n < 3; ++n) {
        globals::s(i, n) = s_init[n];
      }
    }
  }

  bool initial_spin_state_is_a_file = lattice_settings.exists("spins");

  if (initial_spin_state_is_a_file) {
    std::string spin_filename = lattice_settings["spins"];

    cout << "  reading initial spin configuration from " << spin_filename << "\n";

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
    throw general_exception("Lattice::calc_symmetry_operations() was called with symops disabled ", __FILE__, __LINE__, __PRETTY_FUNCTION__);
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
      spg_positions[i][j] = motif_[i].pos[j];
    }
  }

  int (*spg_types) = new int[motif_.size()];

  for (auto i = 0; i < motif_.size(); ++i) {
    spg_types[i] = motif_[i].material;
  }

  spglib_dataset_ = spg_get_dataset(spg_lattice, spg_positions, spg_types, motif_.size(), 1e-5);

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

  int (*primitive_types) = new int[motif_.size()];

  for (auto i = 0; i < motif_.size(); ++i) {
    primitive_types[i] = spg_types[i];
  }

  primitive_num_atoms = spg_find_primitive(primitive_lattice, primitive_positions, primitive_types, motif_.size(), 1e-5);

  // spg_find_primitive returns number of atoms in primitve cell
  if (primitive_num_atoms != motif_.size()) {
    cout << "\n";
    cout << "    !!! unit cell is not a primitive cell !!!\n";
    cout << "\n";
    cout << "    primitive lattice vectors:\n";

    for (int i = 0; i < 3; ++i) {
      cout << "    ";
      cout << primitive_lattice[i][0] << " ";
      cout << primitive_lattice[i][1] << " ";
      cout << primitive_lattice[i][2] << "\n";
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

  if (!lattice->is_periodic(0) && !lattice->is_periodic(1) && !lattice->is_periodic(2)) {
    return std::min(abs(supercell.a()), std::min(abs(supercell.b()), abs(supercell.c()))) / 2.0;
  }

  if (lattice->is_periodic(0) && lattice->is_periodic(1) &&  !lattice->is_periodic(2)) {
    return rhombus_inradius(supercell.a(), supercell.b());
  }

  if (lattice->is_periodic(0) && !lattice->is_periodic(1) &&  lattice->is_periodic(2)) {
    return rhombus_inradius(supercell.a(), supercell.c());
  }

  if (!lattice->is_periodic(0) && lattice->is_periodic(1) &&  lattice->is_periodic(2)) {
    return rhombus_inradius(supercell.a(), supercell.c());
  }

  if (!lattice->is_periodic(0) && !lattice->is_periodic(1) &&  lattice->is_periodic(2)) {
    return abs(supercell.c()) / 2.0;
  }

  if (!lattice->is_periodic(0) && lattice->is_periodic(1) &&  !lattice->is_periodic(2)) {
    return abs(supercell.b()) / 2.0;
  }

  if (lattice->is_periodic(0) && !lattice->is_periodic(1) &&  !lattice->is_periodic(2)) {
    return abs(supercell.a()) / 2.0;
  }

  return 0.0;
}

std::vector<Vec3> Lattice::generate_symmetric_points(const Vec3 &r_cart, const double &tolerance = 1e-6) const {

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

bool Lattice::is_a_symmetry_complete_set(const std::vector<Vec3> &points, const double &tolerance = 1e-6) const {
  for (const auto r : points) {
    for (const auto r_sym : generate_symmetric_points(r, tolerance)) {
      if (!vec_exists_in_container(points, r_sym, tolerance)) {
        return false;
      }
    }
  }
  return true;
}


