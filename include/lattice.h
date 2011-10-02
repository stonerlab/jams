#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <map>
#include <vector>
#include <libconfig.h++>

class Lattice {
  public:
    Lattice() : dim(3,0), nTypes(0), atom_type(0,0), type_count(0), atomTypeMap() {}
    void createFromConfig(libconfig::Config &config);

    inline void getDimensions(int &x, int &y, int& z) { x = dim[0]; y = dim[1]; z = dim[2]; }
    inline int getType(const int i) { return atom_type[i]; }
    inline int getTypeCount(const int i) { return type_count[i]; }
    inline int numTypes() { return nTypes; }
  private:
    void readExchange();

    std::vector<int> dim;
    int nTypes;
    std::vector<int> atom_type;
    std::vector<int> unit_cell_atom_num;
    std::vector<int> type_count;
    std::map<std::string,int> atomTypeMap;
};

#endif // __LATTICE_H__
