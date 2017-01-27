// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_TYPES_H
#define JAMS_CORE_TYPES_H

#include "jblib/containers/vec.h"
#include "jblib/containers/matrix.h"

using std::string;

//-----------------------------------------------------------------------------
// typedefs
//-----------------------------------------------------------------------------

typedef jblib::Vec3<double>          Vec3;
typedef jblib::Vec3<bool>            Vec3b;
typedef jblib::Vec3<int>             Vec3i;

typedef jblib::Matrix<double, 3, 3>  Mat3;

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
