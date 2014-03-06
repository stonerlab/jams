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

    // inline void getDimensions(int &x, int &y, int& z) { x = dim[0]; y = dim[1]; z = dim[2]; }
    // inline void getMaxDimensions(float &x, float& y, float& z) { x = rmax[0]; y = rmax[1]; z = rmax[2]; }
    // inline void getBoundaries(bool &x, bool &y, bool& z) { x = is_periodic[0]; y = is_periodic[1]; z = is_periodic[2]; }
    // inline void getKspaceDimensions(int &x, int &y, int& z) {
    //     x = unitcell_kpoints[0]*dim[0];
    //     y = unitcell_kpoints[1]*dim[1];
    //     z = unitcell_kpoints[2]*dim[2];
    // }


    // inline void getSpinIntCoord(const int &n, int &x, int &y, int &z){
    //     x = spin_int_map(n, 0);
    //     y = spin_int_map(n, 1);
    //     z = spin_int_map(n, 2);
    // }

    inline std::string get_material_name(const int material_number) const {
      return materials_numbered_list_[material_number];
    }

    inline int num_spins_of_material(const int i) const { return material_count_[i]; }
    inline std::string get_material(const int i) const { return lattice_materials_[i]; }
    inline int get_material_number(const int i) { return materials_map_[lattice_materials_[i]]; }
    inline int num_materials() { return materials_map_.size(); }
    void output_spin_state_as_vtu(std::ofstream &outfile);
    void output_spin_state_as_binary(std::ofstream &outfile);
    void output_spin_types_as_binary(std::ofstream &outfile);
    void read_spin_state_from_binary(std::ifstream &infile);
    void initialize_coarse_magnetisation_map();
    void output_coarse_magnetisation(std::ofstream &outfile);
  private:

    void read_lattice(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void compute_positions(const libconfig::Setting &material_settings, const libconfig::Setting &lattice_settings);
    void read_interactions(const libconfig::Setting &lattice_settings);
    void compute_interactions();
    bool insert_interaction(const int i, const int j, const jblib::Matrix<double, 3, 3> &value);

    double energy_cutoff_;

    std::vector<std::string>    materials_numbered_list_;
    std::vector<int>            material_count_;
    std::map<std::string, int>  materials_map_;
    std::vector<std::string>    lattice_materials_;
    double                      lattice_parameter_;
    jblib::Vec3<bool>           lattice_pbc_;
    jblib::Vec3<int>            lattice_size_;
    jblib::Matrix<double, 3, 3> lattice_vectors_;
    jblib::Matrix<double, 3, 3> inverse_lattice_vectors_;
    std::vector< jblib::Vec3<double> > lattice_positions_;
    jblib::Array<int, 4>          fast_integer_lattice_;
    std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > fast_integer_interaction_list_;
    std::vector< std::pair<std::string, jblib::Vec3<double> > > motif_;
};

#endif // JAMS_CORE_LATTICE_H
