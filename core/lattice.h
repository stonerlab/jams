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
    Lattice();
    ~Lattice();

    void init_from_config(const libconfig::Config& pConfig);

    inline double              parameter() const;      ///< lattice parameter (m)
    inline double              volume() const;      ///< volume (m^3)

    inline int                 size(const int i) const;           ///< integer number of unitcell in each lattice vector
    inline int                 num_unit_cells() const; ///< number of unit cells in the whole lattice
    inline int                 num_unit_cell_positions() const;         ///< number atomic positions in the unit cell
    inline jblib::Vec3<double> unit_cell_position(const int i) const;   ///< position i in fractional coordinates
    inline int                 unit_cell_material_uid(const int i);     ///< uid of material at position i
    inline std::string         unit_cell_material_name(const int i);     ///< uid of material at position i

    inline int num_materials() const;                           ///< number of unique materials in lattice
    inline int material_count(const int i) const;               ///< number of lattice sites of material i
    inline int material(const int i);                     ///< uid of material at lattice site i
    inline std::string material_name(const int uid) const;  ///< material string name from uid

    // distance between two lattice sites in cartesian space taking into account the boundary conditions
    double distance(const int i, const int j) const;
    // squared distance between two lattice sites in cartesian space taking into account the boundary conditions
    double distance_sq(const int i, const int j) const;

    inline jblib::Vec3<double> position(const int i) const;
    jblib::Vec3<double> minimum_image(const jblib::Vec3<double> ri, const jblib::Vec3<double> rj) const;
    jblib::Vec3<double> minimum_image_fractional(const jblib::Vec3<double> ri, const jblib::Vec3<double> rj) const;

    inline jblib::Vec3<double> rmax() const;
    inline jblib::Vec3<double> rmin() const;
    jblib::Vec3<double> generate_position(const jblib::Vec3<double> unit_cell_frac_pos, const jblib::Vec3<int> translation_vector) const;
    jblib::Vec3<double> generate_image_position(const jblib::Vec3<double> unit_cell_cart_pos, const jblib::Vec3<int> image_vector) const;
    jblib::Vec3<double> generate_fractional_position(const jblib::Vec3<double> unit_cell_frac_pos, const jblib::Vec3<int> translation_vector) const;


    inline jblib::Vec3<double> cartesian_to_fractional(const jblib::Vec3<double>& r_cart) const;
    inline jblib::Vec3<double> fractional_to_cartesian(const jblib::Vec3<double>& r_frac) const;

    inline int num_sym_ops() const;
    inline jblib::Vec3<double> sym_rotation(const int i, const jblib::Vec3<double> r) const;



    // --------------------------------------------------------------------------
    // lattice vector functions
    // --------------------------------------------------------------------------


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
        assert(i >= 0);
        assert(j < num_unit_cells(1));
        assert(j >= 0);
        assert(k < num_unit_cells(2));
        assert(k >= 0);
        assert(m < num_unit_cell_positions());
        assert(m >= 0);

        return lattice_map_(i, j, k, m);
    }

    inline bool is_bulk_system() const {
        return (lattice_periodic_boundary_.x && lattice_periodic_boundary_.y && lattice_periodic_boundary_.z);
    }

    inline bool is_open_system() const {
        return (!lattice_periodic_boundary_.x && !lattice_periodic_boundary_.y && !lattice_periodic_boundary_.z);
    }

    inline bool is_periodic(const int i) const {
        assert(i < 3);
        return lattice_periodic_boundary_[i];
    }

    bool apply_boundary_conditions(jblib::Vec3<int>& pos) const;
    bool apply_boundary_conditions(jblib::Vec4<int>& pos) const;

    const jblib::Vec3<int>& super_cell_pos(const int i) const {
        return lattice_super_cell_pos_(i);
    }

    // --------------------------------------------------------------------------
    // kspace functions
    // --------------------------------------------------------------------------
    const jblib::Vec3<int>& kspace_size() const {
        return kspace_size_;
    }

    void load_spin_state_from_hdf5(std::string &filename);

  private:
    void init_unit_cell(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void init_lattice_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void init_kspace();
    void calc_symmetry_operations();

    bool is_debugging_enabled_;

    jblib::Matrix<double, 3, 3> unit_cell_;
    jblib::Matrix<double, 3, 3> unit_cell_inverse_;

    std::vector< std::pair<std::string, jblib::Vec3<double> > > unit_cell_position_;

    std::vector<std::string>    material_numbered_list_;
    std::vector<int>            material_count_;
    std::map<std::string, int>  material_map_;
    std::vector<std::string>    lattice_materials_;
    jblib::Vec3<bool>           lattice_periodic_boundary_;
    jblib::Vec3<int>            lattice_size_;
    jblib::Array<jblib::Vec3<int>, 1>        lattice_super_cell_pos_;
    jblib::Array<int, 4>        lattice_map_;

    jblib::Array<int, 3>        kspace_map_;
    jblib::Vec3<int>            unit_cell_kpoints_;
    jblib::Vec3<int>            kpoints_;
    jblib::Vec3<int>            kspace_size_;
    jblib::Array<int, 2>        kspace_inv_map_;
    std::vector< jblib::Vec3<double> > lattice_positions_;
    std::vector< jblib::Vec3<double> > lattice_frac_positions_;
    double                      lattice_parameter_;
    std::vector<int>            lattice_material_num_;
    jblib::Vec3<double>         rmax_;
    jblib::Vec3<double>         rmin_;

    // spglib
    SpglibDataset *spglib_dataset_;

};

inline double
Lattice::parameter() const {
    return lattice_parameter_;
}

inline double
Lattice::volume() const {
    return std::abs(unit_cell_.determinant())*std::pow(lattice_parameter_, 3);
}

inline int
Lattice::size(const int i) const {
    return lattice_size_[i];
}

inline int
Lattice::num_unit_cells() const {
    return product(lattice_size_);
}

inline int
Lattice::num_unit_cell_positions() const {
    return unit_cell_position_.size();
}

inline jblib::Vec3<double>
Lattice::unit_cell_position(const int i) const {
    assert(i < num_unit_cell_positions());
    return unit_cell_position_[i].second;
}

inline int
Lattice::unit_cell_material_uid(const int i) {
    assert(i < unit_cell_position_.size());
    return material_map_[unit_cell_position_[i].first];
}

inline std::string
Lattice::unit_cell_material_name(const int i) {
    assert(i < unit_cell_position_.size());
    return material_name(material_map_[unit_cell_position_[i].first]);
}

inline int
Lattice::num_materials() const {
    return material_map_.size();
}

inline int
Lattice::material_count(const int i) const {
    return material_count_[i];
}

inline int
Lattice::material(const int i) {
    assert(i < lattice_materials_.size());
    return lattice_material_num_[i];
    // return material_map_[lattice_materials_[i]];
}

inline std::string
Lattice::material_name(const int uid) const {
    return material_numbered_list_[uid];
}

inline jblib::Vec3<double>
Lattice::position(const int i) const {
    return lattice_positions_[i];
}

inline jblib::Vec3<double>
Lattice::cartesian_to_fractional(const jblib::Vec3<double>& r_cart) const {
    return unit_cell_inverse_*r_cart;
}

inline jblib::Vec3<double>
Lattice::fractional_to_cartesian(const jblib::Vec3<double>& r_frac) const {
    return unit_cell_*r_frac;
}

inline int
Lattice::num_sym_ops() const {
    return spglib_dataset_->n_operations;
}

inline jblib::Vec3<double>
Lattice::sym_rotation(const int i, const jblib::Vec3<double> vec) const {
    return jblib::Vec3<double>(
    spglib_dataset_->rotations[i][0][0]*vec.x + spglib_dataset_->rotations[i][0][1]*vec.y + spglib_dataset_->rotations[i][0][2]*vec.z,
    spglib_dataset_->rotations[i][1][0]*vec.x + spglib_dataset_->rotations[i][1][1]*vec.y + spglib_dataset_->rotations[i][1][2]*vec.z,
    spglib_dataset_->rotations[i][2][0]*vec.x + spglib_dataset_->rotations[i][2][1]*vec.y + spglib_dataset_->rotations[i][2][2]*vec.z);
}

inline jblib::Vec3<double>
Lattice::rmax() const {
    return rmax_;
};


inline jblib::Vec3<double>
Lattice::rmin() const {
    return rmin_;
};




#endif // JAMS_CORE_LATTICE_H
