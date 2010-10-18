#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <map>
#include <vector>

class Lattice {
  public:
    Lattice() : dim(3,0), ntypes(0), atom_type(0,0), atom_type_map() {}
    void createFromConfig();
  private:
    void readExchange();

    std::vector<int> dim;
    int ntypes;
    std::vector<int> atom_type;
    std::map<std::string,int> atom_type_map;

};

#endif // __LATTICE_H__
