// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_LATTICE_H
#define JAMS_CORE_LATTICE_H

#include <libconfig.h++>

#include <map>
#include <string>
#include <vector>

#include "jblib/containers/array.h"

class Lattice {
  public:
    Lattice() :
        dim(3, 0),
        rmax(3, 0),
        num_atom_types(0),
        atom_type(0, 0),
        unit_cell_atom_num(0, 0),
        type_count(0),
        atom_names(),
        atom_type_map(),
        spin_int_map(0, 0),
        local_atom_pos(0, 0),
        unitcell_kpoints(3, 0),
        lattice_parameter(0.0),
        is_periodic(3, false),
        coarseDim(0),
        spin_to_coarse_cell_map(0, 0),
        coarse_magnetisation_type_count(0, 0, 0, 0),
        coarseMagnetisation(0, 0, 0, 0, 0) {}
    void create_from_config(libconfig::Config &config);

    inline void getDimensions(int &x, int &y, int& z) { x = dim[0]; y = dim[1]; z = dim[2]; }
    inline void getMaxDimensions(float &x, float& y, float& z) { x = rmax[0]; y = rmax[1]; z = rmax[2]; }
    inline void getBoundaries(bool &x, bool &y, bool& z) { x = is_periodic[0]; y = is_periodic[1]; z = is_periodic[2]; }
    inline void getKspaceDimensions(int &x, int &y, int& z) {
        x = unitcell_kpoints[0]*dim[0];
        y = unitcell_kpoints[1]*dim[1];
        z = unitcell_kpoints[2]*dim[2];
    }


    inline void getSpinIntCoord(const int &n, int &x, int &y, int &z){
        x = spin_int_map(n, 0);
        y = spin_int_map(n, 1);
        z = spin_int_map(n, 2);
    }

    inline int getType(const int i) { return atom_type[i]; }
    inline int getTypeCount(const int i) { return type_count[i]; }
    inline int numTypes() { return num_atom_types; }
    void output_spin_state_as_vtu(std::ofstream &outfile);
    void output_spin_state_as_binary(std::ofstream &outfile);
    void output_spin_types_as_binary(std::ofstream &outfile);
    void read_spin_state_from_binary(std::ifstream &infile);
    void initialize_coarse_magnetisation_map();
    void output_coarse_magnetisation(std::ofstream &outfile);
  private:
    void readExchange();
    void calculateAtomPos(const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, jblib::Array<int, 4> &latt, std::vector<int> &dim, const double unitcell[3][3], const int num_atoms);
    void map_position_to_int();
    void readLattice(const libconfig::Setting &cfgLattice, std::vector<int> &dim, bool pbc[3], const double unitcel[3][3]);

    std::vector<int> dim;
    std::vector<float> rmax;
    int num_atom_types;
    std::vector<int> atom_type;
    std::vector<int> unit_cell_atom_num;
    std::vector<int> type_count;
    std::vector<std::string> atom_names;
    std::map<std::string, int> atom_type_map;
    jblib::Array<int, 2>     spin_int_map;
    jblib::Array<double, 2>     local_atom_pos;
    std::vector<int> unitcell_kpoints;
    float lattice_parameter;
    std::vector<bool> is_periodic;
    std::vector<int> coarseDim;
    jblib::Array<int, 2> spin_to_coarse_cell_map;
    jblib::Array<int, 4> coarse_magnetisation_type_count;
    jblib::Array<double, 5> coarseMagnetisation;
};

#endif // JAMS_CORE_LATTICE_H
