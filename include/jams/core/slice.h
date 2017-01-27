// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SLICE_H
#define JAMS_CORE_SLICE_H

#include <fstream>
#include <vector>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/utils.h"

#include "jblib/containers/array.h"

class Slice {
 public:

  Slice()
    : num_slice_points(0),
      slice_origin(0.0, 0.0, 0.0),
      slice_size(0.0, 0.0, 0.0),
      slice_map() {}

  explicit Slice(const libconfig::Setting &settings)
    : num_slice_points(0),
      slice_origin(0.0, 0.0, 0.0),
      slice_size(0.0, 0.0, 0.0),
      slice_map() {
    using namespace globals;
    using jblib::floats_are_greater_than_or_equal;
    using jblib::floats_are_less_than_or_equal;

    for (int i = 0; i < 3; ++i) {
      slice_origin[i] = settings["origin"][i];
    }
    ::output.write("  slice origin: %f %f %f\n", slice_origin[0], slice_origin[1], slice_origin[2]);
    for (int i = 0; i < 3; ++i) {
      slice_size[i] = settings["size"][i];
    }
    ::output.write("  slice size: %f %f %f\n", slice_size[0], slice_size[1], slice_size[2]);

    for (int i = 0; i < num_spins; ++i) {
      jblib::Vec3<double> pos = lattice.atom_position(i);

            // check if the current spin in inside the slice
      if (floats_are_greater_than_or_equal(pos.x, slice_origin.x) && floats_are_less_than_or_equal(pos.x, slice_origin.x + slice_size.x)
        &&  floats_are_greater_than_or_equal(pos.y, slice_origin.y) && floats_are_less_than_or_equal(pos.y, slice_origin.y + slice_size.y)
        &&  floats_are_greater_than_or_equal(pos.z, slice_origin.z) && floats_are_less_than_or_equal(pos.z, slice_origin.z + slice_size.z)) {
        slice_map.push_back(i);
      }
    }

    num_slice_points = slice_map.size();
  }

  ~Slice() {}

  int num_points() {
    return num_slice_points;
  }

  int& index(const int i) {
    assert(i < num_slice_points);
    return slice_map[i];
  }

  double& spin(const int i, const int j) {
    return globals::s(slice_map[i], j);
  }

  double position(const int i, const int j) {
    return lattice.parameter()*lattice.atom_position(slice_map[i])[j];
  }

  int type(const int i) {
    return lattice.atom_material(slice_map[i]);
  }

 private:
    int                 num_slice_points;
    jblib::Vec3<double> slice_origin;
    jblib::Vec3<double> slice_size;
    std::vector<int>    slice_map;
};

#endif  // JAMS_CORE_SLICE_H

