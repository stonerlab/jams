//
// Created by Joseph Barker on 2019-04-23.
//

#ifndef JAMS_H5_H
#define JAMS_H5_H

#include <highfive/H5File.hpp>
#include <highfive/bits/H5Utils.hpp>
#include <highfive/bits/H5Converter_misc.hpp>

// add support to HighFive for our own MultiArray type
namespace HighFive {
    namespace details {

        template<class Tp_, std::size_t Dim_, class Idx_>
        struct type_of_array<jams::MultiArray<Tp_, Dim_, Idx_>> {
        typedef typename type_of_array<Tp_>::type type;
    };

    template<class Tp_, std::size_t Dim_, class Idx_>
    struct array_dims<jams::MultiArray<Tp_, Dim_, Idx_>> {
    static constexpr size_t value = Dim_;
};

// apply conversion to jams::MultiArray
template<class Tp_, std::size_t Dim_, class Idx_>
struct data_converter<jams::MultiArray<Tp_, Dim_, Idx_>, void> {

typedef typename jams::MultiArray<Tp_, Dim_, Idx_> MultiArray;

inline data_converter(MultiArray&, DataSpace& space)
    : _dims(space.getDimensions()) {
  assert(_dims.size() == Dim_);
}

inline typename type_of_array<Tp_>::type* transform_read(MultiArray& array) {
  if (std::equal(_dims.begin(), _dims.end(), std::begin(array.shape())) == false) {
    std::array<Idx_, Dim_> ext;
    std::copy(_dims.begin(), _dims.end(), ext.begin());
    array.resize(ext);
  }
  return array.data();
}

inline typename type_of_array<Tp_>::type* transform_write(MultiArray& array) {
  return array.data();
}

inline void process_result(MultiArray&) {}

std::vector<size_t> _dims;
};
}
}

#endif //JAMS_H5_H
