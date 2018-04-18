//
// Created by Joe Barker on 2017/09/14.
//

#ifndef JAMS_LOAD_H
#define JAMS_LOAD_H

#include <vector>
#include <jblib/containers/array.h>
#include "H5Cpp.h"
#include "utils.h"

template <int N>
void load_array_from_h5_file(const std::string& file_name, const std::string& data_set_path, jblib::Array<double, N>& array) {
  using namespace H5;

  H5File file(file_name.c_str(), H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet(data_set_path);
  DataSpace dataspace = dataset.getSpace();

  if (dataspace.getSimpleExtentNpoints() != static_cast<hssize_t>(array.elements())) {
    throw std::runtime_error(
            "loading array from file: '" + file_name + "' expected size: " + std::to_string(array.elements()) +
            " actual size: " + std::to_string(dataspace.getSimpleExtentNpoints())
    );
  }

  dataset.read(array.data(), PredType::NATIVE_DOUBLE);
}

template <int N>
void load_array_from_tsv_file(const std::string& file_name, jblib::Array<double, N> array) {
  std::ifstream tsv_file(file_name);

  if(!tsv_file.is_open()) {
    throw std::runtime_error("failed to open file: " + file_name);
  }

  std::size_t data_count = 0;

  for (std::string line; getline(tsv_file, line); ) {
    if (line.empty() || string_is_comment(line)) {
      continue;
    }

    std::stringstream is(line);
    while(is.good()) {
      is >> array[data_count];
      data_count++;
    }
  }

  if (array.elements() != data_count) {
    throw std::runtime_error(
            "loading array from file: '" + file_name + "' expected size: " + std::to_string(array.elements()) +
            " actual size: " + std::to_string(data_count)
    );
  }
}

template <int N>
void load_array_from_file(const std::string& file_name, const std::string& data_set_path, jblib::Array<double, N>& array) {
  using namespace H5;
  if (H5File::isHdf5(file_name)) {
    load_array_from_h5_file(file_name, data_set_path, array);
  } else {
    load_array_from_tsv_file(file_name, array);
  }
}


#endif //JAMS_LOAD_H
