// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

extern "C"{
    #include "spglib/spglib.h"
}

#include <libconfig.h++>

#include <map>
#include <string>
#include <vector>
#include <complex>

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

class Lattice {
  public:
    Lattice()  {}
    void initialize(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);

    // --------------------------------------------------------------------------
    // material functions
    // --------------------------------------------------------------------------
    inline int
    num_materials() const {
        return materials_map_.size();
    }

    inline std::string
    material(const int i) const {
        return lattice_materials_[i];
    }

    inline int
    material_id(const int i) {
        assert(i < lattice_materials_.size());
        return materials_map_[lattice_materials_[i]];
    }

    inline std::string
    material_name(const int material_number) const {
      return materials_numbered_list_[material_number];
    }

    inline int
    num_spins_of_material(const int i) const {
        return material_count_[i];
    }

    // --------------------------------------------------------------------------
    // motif functions
    // --------------------------------------------------------------------------

    inline int
    num_unit_cell_positions() const {
        return unit_cell_positions_.size();
    }

    inline jblib::Vec3<double>
    unit_cell_position(const int i) const {
        assert(i < num_unit_cell_positions_());
        return unit_cell_positions_[i].second;
    }

    inline std::string
    unit_cell_material(const int i) const {
        assert(i < num_unit_cell_positions_());
        return unit_cell_positions_[i].first;
    }

    inline int
    unit_cell_material_id(const int i) {
        return materials_map_[unit_cell_material(i)];
    }

    // --------------------------------------------------------------------------
    // lattice vector functions
    // --------------------------------------------------------------------------

    inline jblib::Vec3<double> cartesian_to_fractional_position(const jblib::Vec3<double>& r_cart) {
        return unit_cell_inverse_*r_cart;
    }

    inline jblib::Vec3<double> fractional_to_cartesian_position(const jblib::Vec3<double>& r_frac) {
        return unit_cell_*r_frac;
    }

    inline const jblib::Vec3<double>& position(const int i) const {
        return lattice_positions_[i];
    }

    inline double constant() const {
        return lattice_parameter_;
    }

    // --------------------------------------------------------------------------
    // lattice vector functions
    // --------------------------------------------------------------------------
    inline int num_unit_cells(const int i) const {
        assert(i < 3);
        return lattice_size_[i];
    }

    // lookup the site index but unit cell integer coordinates and motif offset
    inline int site_index_by_unit_cell(const int i, const int j, const int k, const int m) const {
        assert(i < num_unit_cells(0));
        assert(j < num_unit_cells(1));
        assert(k < num_unit_cells(2));
        assert(m < num_unit_cell_positions_());
        return lattice_integer_lookup_(i, j, k, m);
    }

    inline bool is_periodic(const int i) const {
        assert(i < 3);
        return lattice_pbc_[i];
    }

    bool apply_boundary_conditions(jblib::Vec3<int>& pos) const;
    bool apply_boundary_conditions(jblib::Vec4<int>& pos) const;


    // --------------------------------------------------------------------------
    // symmetry functions
    // --------------------------------------------------------------------------

    inline int num_sym_ops() const {
        return spglib_dataset_->n_operations;
    }

    inline jblib::Matrix<int, 3, 3> sym_rotation(const int i) const {
         return jblib::Matrix<int, 3, 3>(
            spglib_dataset_->rotations[i][0][0], spglib_dataset_->rotations[i][0][1], spglib_dataset_->rotations[i][0][2],
            spglib_dataset_->rotations[i][1][0], spglib_dataset_->rotations[i][1][1], spglib_dataset_->rotations[i][1][2],
            spglib_dataset_->rotations[i][2][0], spglib_dataset_->rotations[i][2][1], spglib_dataset_->rotations[i][2][2]);
    }

    const jblib::Vec3<int>& super_cell_pos(const int i) const {
        return lattice_super_cell_pos_(i);
    }

    // --------------------------------------------------------------------------
    // kspace functions
    // --------------------------------------------------------------------------
    const jblib::Vec3<int>& kspace_size() const {
        return kspace_size_;
    }

    void calculate_unit_cell_kpoints();

    jblib::Array<int, 2>        kspace_inv_map_;
    std::vector< jblib::Vec3<double> > lattice_positions_;
    std::vector< jblib::Vec3<double> > lattice_frac_positions_;
    double                      lattice_parameter_;
    std::vector<int>            lattice_material_num_;
    jblib::Vec3<double>         rmax;
    jblib::Vec3<double>         rmin;

    void load_spin_state_from_hdf5(std::string &filename);

  private:
    void ReadConfig(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void CalculatePositions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void CalculateKspace();
    void CalculateSymmetryOperations();

    bool is_debugging_enabled_;

    jblib::Matrix<double, 3, 3> unit_cell_;
    jblib::Matrix<double, 3, 3> unit_cell_inverse_;

    std::vector< std::pair<std::string, jblib::Vec3<double> > > unit_cell_positions_;

    std::vector<std::string>    materials_numbered_list_;
    std::vector<int>            material_count_;
    std::map<std::string, int>  materials_map_;
    std::vector<std::string>    lattice_materials_;
    jblib::Vec3<bool>           lattice_pbc_;
    jblib::Vec3<int>            lattice_size_;
    jblib::Array<jblib::Vec3<int>, 1>        lattice_super_cell_pos_;
    jblib::Array<int, 4>        lattice_integer_lookup_;

    jblib::Array<int, 3>        kspace_map_;
    jblib::Vec3<int>            unit_cell_kpoints_;
    jblib::Vec3<int>            kpoints_;
    jblib::Vec3<int>            kspace_size_;

    // spglib
    SpglibDataset *spglib_dataset_;

};

#endif // JAMS_CORE_LATTICE_H
