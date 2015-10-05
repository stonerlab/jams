#include <exception>
#include <string>

#include "H5Cpp.h"

#include "hdf5.h"

#include "jblib/containers/array.h";

template <int dim>
void load_array_data_from_hdf5(const std::string &filename,
                               const std::string &dataset_name,
                               const jblib::Array<double, dim> array) {
  using namespace H5;

  H5File file(filename.c_str(), H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet(dataset_name.c_str());
  DataSpace dataspace = dataset.getSpace();

  if (dataspace.getSimpleExtentNpoints() != static_cast<hssize_t>(array.elements()){
    throw std::runtime_exception("Cannot load data from hdf5 file (" + filename.c_str() + "), "
                                 "array size ( " + array.elements() + ") "
                                 "does not match file data size (" + dataspace.getSimpleExtentNpoints() + ")");
  }

  dataset.read(array.data(), PredType::NATIVE_DOUBLE);
}

template <int dim>
void load_array_data_from_hdf5(const std::string &filename,
                               const std::string &dataset_name,
                               const jblib::Array<int, dim> array) {
  using namespace H5;

  H5File file(filename.c_str(), H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet(dataset_name.c_str());
  DataSpace dataspace = dataset.getSpace();

  if (dataspace.getSimpleExtentNpoints() != static_cast<hssize_t>(array.elements()){
    throw std::runtime_exception("Cannot load data from hdf5 file (" + filename.c_str() + "), "
                                 "array size ( " + array.elements() + ") "
                                 "does not match file data size (" + dataspace.getSimpleExtentNpoints() + ")");
  }

  dataset.read(array.data(), PredType::NATIVE_INT);
}
