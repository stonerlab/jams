// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

#include <libconfig.h++>

#include <map>
#include <string>
#include <vector>

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

class Lattice {
  public:
    Lattice()  {}
    void initialize();

    // material function
    inline int num_materials() const {
        return materials_map_.size();
    }
    inline std::string material(const int i) const {
        return lattice_materials_[i];
    }
    inline int material_id(const int i) {
        return materials_map_[lattice_materials_[i]];
    }
    inline std::string material_name(const int material_number) const {
      return materials_numbered_list_[material_number];
    }
    inline int num_spins_of_material(const int i) const {
        return material_count_[i];
    }

    // motif functions
    inline int num_motif_positions() const {
        return motif_.size();
    }

    inline jblib::Vec3<double> motif_position(const int i) const {
        return motif_[i].second;
    }
    inline std::string motif_material(const int i) const {
        return motif_[i].first;
    }
    inline int motif_material_id(const int i) {
        return materials_map_[motif_material(i)];
    }

    // lattice vector functions
    inline jblib::Vec3<double> cartesian_to_fractional_position(const jblib::Vec3<double>& r_cart) {
        return inverse_lattice_vectors_*r_cart;
    }

    inline jblib::Vec3<double> fractional_to_cartesian_position(const jblib::Vec3<double>& r_frac) {
        return lattice_vectors_*r_frac;
    }

    void output_spin_state_as_vtu(std::ofstream &outfile);
    void output_spin_state_as_binary(std::ofstream &outfile);
    void output_spin_types_as_binary(std::ofstream &outfile);
    void read_spin_state_from_binary(std::ifstream &infile);
    void initialize_coarse_magnetisation_map();
    void output_coarse_magnetisation(std::ofstream &outfile);
    jblib::Array<int, 2>        kspace_inv_map_;
    std::vector< jblib::Vec3<double> > lattice_positions_;
    double                      lattice_parameter_;
    std::vector<int>            lattice_material_num_;
    jblib::Vec3<double>         rmax;

    inline int kspace_size(const int i) const { assert(i >= 0 && i < 3); return kspace_size_[i]; }

    void load_spin_state_from_hdf5(std::string &filename);

  private:
    void calculate_unit_cell_kmesh();
    void read_lattice(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void compute_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void read_interactions(const libconfig::Setting &lattice_settings);
    void compute_exchange_interactions();
    void compute_fft_exchange_interactions();
    void compute_fft_dipole_interactions();
    bool insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value);

    double energy_cutoff_;

    std::vector<std::string>    materials_numbered_list_;
    std::vector<int>            material_count_;
    std::map<std::string, int>  materials_map_;
    std::vector<std::string>    lattice_materials_;
    jblib::Vec3<bool>           lattice_pbc_;
    jblib::Vec3<int>            lattice_size_;
    jblib::Matrix<double, 3, 3> lattice_vectors_;
    jblib::Matrix<double, 3, 3> inverse_lattice_vectors_;
    jblib::Array<int, 4>          fast_integer_lattice_;
    std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > fast_integer_interaction_list_;
    std::vector< std::pair<std::string, jblib::Vec3<double> > > motif_;
    jblib::Array<int, 3>        kspace_map_;
    jblib::Vec3<int>            kpoints_;
    jblib::Vec3<int>            kspace_size_;
};

#endif // JAMS_CORE_LATTICE_H
