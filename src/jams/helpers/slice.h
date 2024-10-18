// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SLICE_H
#define JAMS_CORE_SLICE_H

#include <fstream>
#include <vector>
#include <iostream>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "utils.h"

class Slice {
 public:

  Slice()
    : num_slice_points(0),
      slice_origin{0.0, 0.0, 0.0},
      slice_size{0.0, 0.0, 0.0},
      slice_map() {}

  explicit Slice(const libconfig::Setting &settings)
    : num_slice_points(0),
      slice_origin{0.0, 0.0, 0.0},
      slice_size{0.0, 0.0, 0.0},
      slice_map() {
    for (int i = 0; i < 3; ++i) {
      slice_origin[i] = settings["origin"][i];
    }
    std::cout << "  slice origin: " << slice_origin[0] << " " << slice_origin[1] << " " << slice_origin[2] << "\n";

    for (int i = 0; i < 3; ++i) {
      slice_size[i] = settings["size"][i];
    }
    std::cout << "  slice size: " << slice_size[0] << " " << slice_size[1] << " " << slice_size[2] << "\n";

    for (int i = 0; i < globals::num_spins; ++i) {
      Vec3 pos = globals::lattice->lattice_site_position_cart(i);

            // check if the current spin in inside the slice
      if (definately_greater_than(pos[0], slice_origin[0], jams::defaults::lattice_tolerance) && definately_less_than(pos[0], slice_origin[0] + slice_size[0], jams::defaults::lattice_tolerance)
        &&  definately_greater_than(pos[1], slice_origin[1], jams::defaults::lattice_tolerance) && definately_less_than(pos[1], slice_origin[1] + slice_size[1], jams::defaults::lattice_tolerance)
        &&  definately_greater_than(pos[2], slice_origin[2], jams::defaults::lattice_tolerance) && definately_less_than(pos[2], slice_origin[2] + slice_size[2], jams::defaults::lattice_tolerance)) {
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
    return globals::lattice->parameter() *
        globals::lattice->lattice_site_position_cart(slice_map[i])[j];
  }

  int type(const int i) {
    return globals::lattice->lattice_site_material_id(slice_map[i]);
  }

 private:
    int                 num_slice_points;
    Vec3 slice_origin;
    Vec3 slice_size;
    std::vector<int>    slice_map;
};

#endif  // JAMS_CORE_SLICE_H

