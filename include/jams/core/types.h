// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_TYPES_H
#define JAMS_CORE_TYPES_H

#include "jams/containers/vec3.h"
#include "jams/containers/mat3.h"

using std::string;

//-----------------------------------------------------------------------------
// enums
//-----------------------------------------------------------------------------

enum class CoordinateFormat {Cartesian, Fractional};

//-----------------------------------------------------------------------------
// typedefs
//-----------------------------------------------------------------------------

using Mat3  = std::array<std::array<double, 3>, 3>;

using Vec3  = std::array<double, 3>;
using Vec3b = std::array<bool, 3>;
using Vec3i = std::array<int, 3>;

using Vec4  = std::array<double, 4>;
using Vec4i = std::array<int, 4>;

//-----------------------------------------------------------------------------
// structs
//-----------------------------------------------------------------------------

struct Atom {
    int  id;
    int  material;
    Vec3 pos;
};

struct SuperCell {
    Vec3i  size;
    Vec3b  periodic;
    Mat3   unit_cell;
    Mat3   unit_cell_inv;
    double parameter;
};

struct Material {
    int    id;
    string name;
    double moment;
    double gyro;
    double alpha;
    Vec3   spin;
    Vec3   transform;
    bool   randomize;
};


#endif  // JAMS_CORE_TYPES_H
