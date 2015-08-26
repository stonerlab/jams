// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_HDF5_H
#define JAMS_CORE_HDF5_H

#include <string>

#include "H5Cpp.h"

#include "jblib/containers/array.h";

template <int dim>
void load_array_data_from_hdf5(const std::string &filename,
                               const std::string &dataset_name,
                               const jblib::Array<double, dim> array);
template <int dim>
void load_array_data_from_hdf5(const std::string &filename,
                               const std::string &dataset_name,
                               const jblib::Array<int, dim> array);
#endif  // JAMS_CORE_HDF5_H