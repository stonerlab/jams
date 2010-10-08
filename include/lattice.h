#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <map>
#include <vector>

// struct for convenience
template <typename _Tp>
struct vec3
{
  public:
    vec3(){};
    vec3(const _Tp x, const _Tp y, const _Tp z)
      {data[0]=x;data[1]=y;data[2]=z;}
    _Tp RESTRICT &operator[](const int i){return data[i];}
    const _Tp &operator[](const int i) const {return data[i];}
  private:
    _Tp data[3];
};


class Lattice {
  public:
    void createFromConfig();
  private:
    void readExchange();

    int      dim[3];

    int ntypes;
    std::vector<int> atom_type;
    std::map<std::string,int> atom_type_map;

};

#endif // __LATTICE_H__
