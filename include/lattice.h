#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <map>
#include <vector>
#include <libconfig.h++>
#include "array.h"
#include "array2d.h"
#include "array4d.h"
#include "array5d.h"

class Lattice {
  public:
    Lattice() : 
        dim(3,0), 
        rmax(3,0), 
        nTypes(0), 
        atom_type(0,0), 
        unit_cell_atom_num(0,0),
        type_count(0), 
        atomNames(),
        atomTypeMap(), 
        spin_int_map(0,0),
        local_atom_pos(0,0),
        unitcell_kpoints(3,0), 
        latticeParameter(0.0), 
        boundaries(3,false),
        coarseDim(0),
        spinToCoarseMap(0,0),
        coarseMagnetisationTypeCount(0,0,0,0),
        coarseMagnetisation(0,0,0,0,0) {}
    void createFromConfig(libconfig::Config &config);

    inline void getDimensions(int &x, int &y, int& z) { x = dim[0]; y = dim[1]; z = dim[2]; }
    inline void getMaxDimensions(float &x, float& y, float& z) { x = rmax[0]; y = rmax[1]; z = rmax[2]; }
    inline void getBoundaries(bool &x, bool &y, bool& z) { x = boundaries[0]; y = boundaries[1]; z = boundaries[2]; }
    inline void getKspaceDimensions(int &x, int &y, int& z) {
        x = unitcell_kpoints[0]*dim[0];
        y = unitcell_kpoints[1]*dim[1];
        z = unitcell_kpoints[2]*dim[2];
    }


    inline void getSpinIntCoord(const int &n, int &x, int &y, int &z){
        x = spin_int_map(n,0);
        y = spin_int_map(n,1);
        z = spin_int_map(n,2);
    }

    inline int getType(const int i) { return atom_type[i]; }
    inline int getTypeCount(const int i) { return type_count[i]; }
    inline int numTypes() { return nTypes; }
    void outputSpinsVTU(std::ofstream &outfile);
    void outputSpinsBinary(std::ofstream &outfile);
    void outputTypesBinary(std::ofstream &outfile);
    void readSpinsBinary(std::ifstream &infile);
    void initializeCoarseMagnetisationMap();
    void outputCoarseMagnetisationMap(std::ofstream &outfile);
  private:
    void readExchange();
    void calculateAtomPos(const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array4D<int> &latt, std::vector<int> &dim, const double unitcell[3][3], const int nAtoms);
    void mapPosToInt();
    void readLattice(const libconfig::Setting &cfgLattice, std::vector<int> &dim, bool pbc[3], const double unitcel[3][3]);

    std::vector<int> dim;
    std::vector<float> rmax;
    int nTypes;
    std::vector<int> atom_type;
    std::vector<int> unit_cell_atom_num;
    std::vector<int> type_count;
    std::vector<std::string> atomNames;
    std::map<std::string,int> atomTypeMap;
    Array2D<int>     spin_int_map;
    Array2D<double>     local_atom_pos;
    std::vector<int> unitcell_kpoints;
    float latticeParameter;
    std::vector<bool> boundaries;
    std::vector<int> coarseDim;
    Array2D<int> spinToCoarseMap;
    Array4D<int> coarseMagnetisationTypeCount;
    Array5D<double> coarseMagnetisation;
};

#endif // __LATTICE_H__
