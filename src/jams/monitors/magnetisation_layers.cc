// magnetisation_layers.cc                                             -*-C++-*-
#include <jams/monitors/magnetisation_layers.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>

#include <jams/helpers/maths.h>
#include <jams/helpers/spinops.h>
#include <jams/interface/highfive.h>
#include <jams/helpers/output.h>

#include <map>
#include "jams/cuda/cuda_spin_ops.h"


MagnetisationLayersMonitor::MagnetisationLayersMonitor(
    const libconfig::Setting &settings)
    : Monitor(settings) {

  Vec3 norm = jams::config_required<Vec3>(settings, "norm");

  // construct a rotation matrix which will rotate the system so that the norm
  // is always along z
  Mat3 rotation_matrix = rotation_matrix_between_vectors(norm, Vec3{0, 0, 1});

  // find the unique layer positions
  auto comp_less = [&](double a, double b) -> bool {
      return definately_less_than(a, b, jams::defaults::lattice_tolerance);
  };

  std::map<double, std::vector<int>, decltype(comp_less)> unique_positions(comp_less);

  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 r = rotation_matrix * ::lattice->atom_position(i);
    unique_positions[r[2]].push_back(i);
  }

  num_layers_ = unique_positions.size();


  layer_spin_indicies_.resize(num_layers_);
  layer_magnetisation_.resize(num_layers_, 3);

  jams::MultiArray<double,1> layer_positions(num_layers_);
  jams::MultiArray<int,1> layer_spin_count(num_layers_);

  int counter = 0;
  for (auto const& x : unique_positions) {
    layer_positions(counter) = x.first;
    layer_spin_count(counter) = x.second.size();
    layer_spin_indicies_[counter].resize(x.second.size());

    for (auto i = 0; i < x.second.size(); ++i) {
      layer_spin_indicies_[counter](i) = x.second[i];
    }
    counter++;
  }

  HighFive::File file(jams::output::full_path_filename("monitors.h5"),
                      HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

  HighFive::Group group = file.createGroup("/jams/monitors/magnetisation_layers/");
  group.createAttribute<double>("norm", HighFive::DataSpace(3)).write(norm);
  {
    auto dataset = group.createDataSet<double>(
        "layer_positions",HighFive::DataSpace::From(layer_positions));
    dataset.write(layer_positions);
  }
  {
    auto dataset = group.createDataSet<int>(
        "layer_spin_count",HighFive::DataSpace::From(layer_spin_count));
    dataset.write(layer_spin_count);
  }
}


void MagnetisationLayersMonitor::update(Solver *solver) {
  using namespace HighFive;

  for (auto layer_index = 0; layer_index < num_layers_; ++layer_index) {
    Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus,
                                         layer_spin_indicies_[layer_index]);
    layer_magnetisation_(layer_index, 0) = mag[0];
    layer_magnetisation_(layer_index, 1) = mag[1];
    layer_magnetisation_(layer_index, 2) = mag[2];
  }

  File file(jams::output::full_path_filename("monitors.h5"), File::ReadWrite);

  Group group = file.createGroup("/jams/monitors/magnetisation_layers/" + zero_pad_number(solver->iteration(),9));
  group.createAttribute<double>("timestep", DataSpace(1)).write(
      solver->time());
  auto dataset = group.createDataSet<double>("layer_magnetisation", DataSpace::From(layer_magnetisation_));
  dataset.write(layer_magnetisation_);
}
