//
// Created by Joe Barker on 2017/09/14.
//

#ifndef JAMS_LOAD_H
#define JAMS_LOAD_H

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "jams/containers/multiarray.h"
#include "jams/interface/highfive.h"

namespace jams::detail {

template <typename T>
constexpr const char* h5_float_target_name() {
  if constexpr (std::is_same_v<T, float>) {
    return "Float32";
  } else if constexpr (std::is_same_v<T, double>) {
    return "Float64";
  } else {
    return "floating-point";
  }
}

template <std::size_t N, typename Source, typename Target>
void copy_h5_array_cast(
    const HighFive::DataSet& dataset,
    jams::MultiArray<Target, N>& array) {
  jams::MultiArray<Source, N> source;
  dataset.read(source);

  array.resize(source.shape());
  const auto source_values = source.host_span();
  auto target_values = array.mutable_host_span();

  std::transform(
      source_values.begin(),
      source_values.end(),
      target_values.begin(),
      [](const Source value) {
        return static_cast<Target>(value);
      });
}

inline std::string h5_float_type_error(
    const std::string& filename,
    const std::string& data_set_path,
    const HighFive::DataType& data_type,
    const std::string& target_type) {
  std::ostringstream message;
  message << "failed to load H5 dataset '" << data_set_path << "' from '"
          << filename << "': unsupported source type " << data_type.string()
          << "; expected Float32 or Float64 dataset for target " << target_type;
  return message.str();
}

}  // namespace jams::detail

template <std::size_t N, typename T>
void load_array_from_h5_file(const std::string& filename, const std::string& data_set_path, jams::MultiArray<T, N>& array) {
  using namespace HighFive;
  File file(filename, File::ReadOnly);
  auto dataset = file.getDataSet(data_set_path);
  const auto data_type = dataset.getDataType();

  if constexpr (std::is_floating_point_v<T>) {
    if (data_type.getClass() != DataTypeClass::Float) {
      throw std::runtime_error(jams::detail::h5_float_type_error(
          filename, data_set_path, data_type, jams::detail::h5_float_target_name<T>()));
    }

    if (data_type.getSize() == sizeof(float)) {
      jams::detail::copy_h5_array_cast<N, float>(dataset, array);
      return;
    }

    if (data_type.getSize() == sizeof(double)) {
      jams::detail::copy_h5_array_cast<N, double>(dataset, array);
      return;
    }

    throw std::runtime_error(jams::detail::h5_float_type_error(
        filename, data_set_path, data_type, jams::detail::h5_float_target_name<T>()));
  } else {
    dataset.read(array);
  }
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
