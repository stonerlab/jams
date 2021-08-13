//
// Created by Joseph Barker on 2019-04-23.
//

#ifndef JAMS_HIGHFIVE_H
#define JAMS_HIGHFIVE_H

#include <highfive/H5File.hpp>
#include <highfive/bits/H5Utils.hpp>
#include <highfive/bits/H5Converter_misc.hpp>

// add support to HighFive for our own MultiArray type
namespace HighFive {
    namespace details {

    template<class Tp_, std::size_t Dim_, class Idx_>
    struct inspector<jams::MultiArray<Tp_, Dim_, Idx_>> {
        using type = jams::MultiArray<Tp_, Dim_, Idx_>;
        using value_type = Tp_;
        using base_type = typename inspector<value_type>::base_type;

        static constexpr size_t ndim = Dim_;
        static constexpr size_t recursive_ndim = ndim + inspector<value_type>::recursive_ndim;

        static std::array<size_t, recursive_ndim> getDimensions(const type& val) {
          std::array<size_t, recursive_ndim> sizes{val.shape()};
          return sizes;
        }
    };

// apply conversion to jams::MultiArray
template<class Tp_, std::size_t Dim_, class Idx_>
struct data_converter<jams::MultiArray<Tp_, Dim_, Idx_>, void>
    : public container_converter<jams::MultiArray<Tp_, Dim_, Idx_>> {


    using MultiArray = jams::MultiArray<Tp_, Dim_, Idx_>;
    using value_type = typename inspector<Tp_>::base_type;
    using container_converter<MultiArray>::container_converter;

  inline value_type* transform_read(MultiArray& array) {
    if (std::equal(_dims.begin(), _dims.end(), std::begin(array.shape())) == false) {
      std::array<Idx_, Dim_> ext;
      std::copy(_dims.begin(), _dims.end(), ext.begin());
      array.resize(ext);
    }
    return array.data();
  }

  inline const value_type* transform_write(const MultiArray& array) const noexcept{
    return array.data();
  }

  std::vector<size_t> _dims;
};
}
}

#endif //JAMS_HIGHFIVE_H
