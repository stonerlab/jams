//
// Created by Joe Barker on 2017/09/14.
//

#ifndef JAMS_LOAD_H
#define JAMS_LOAD_H

#include <fstream>
#include <vector>
#include "jams/containers/multiarray.h"
#include "jams/interface/highfive.h"

template <std::size_t N, typename T>
void load_array_from_h5_file(const std::string& filename, const std::string& data_set_path, jams::MultiArray<T, N>& array) {
  using namespace HighFive;
  File file(filename, File::ReadOnly);
  auto dataset = file.getDataSet(data_set_path);
  dataset.read(array);
}

template <std::size_t N, typename T>
void load_array_from_tsv_file(const std::string& file_name, jams::MultiArray<T, N>& array) {
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
      is >> array.data()[data_count];
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

template <std::size_t N, typename T>
void load_array_from_file(const std::string& file_name, const std::string& data_set_path, jams::MultiArray<T, N>& array) {
  if (file_extension(file_name) == "h5") {
    load_array_from_h5_file(file_name, data_set_path, array);
  } else {
    load_array_from_tsv_file(file_name, array);
  }
}


#endif //JAMS_LOAD_H
