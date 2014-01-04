#ifndef JAMS_CORE_GEOMETRY_H
#define JAMS_CORE_GEOMETRY_H

#include <vector>
#include <cstring>
#include <map>

// struct for convenience
template <typename T>
struct vec3
{
  public:
    vec3(){};
    vec3(const T x, const T y, const T z)
      {_data[0]=x;_data[1]=y;_data[2]=z;}
    T &operator[](const int i){return _data[i];}
    const T &operator[](const int i) const {return _data[i];}
  private:
    T _data[3];
};

class Geometry {
  public:
    void readFromConfig();
  private:
    // lattice vectors
    float a0[3];
    float a1[3];
    float a2[3];

    int lattice_size[3];
    int lattice_pbc[3];

    std::vector< vec3<float> >  r;
    std::vector<int>            atom_type;
    std::map<std::string,int>   atom_type_map;
    double  energy_cutoff;

};

#endif // JAMS_CORE_GEOMETRY_H
