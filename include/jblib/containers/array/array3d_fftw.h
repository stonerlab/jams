// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_ARRAY3D_FFTW_H
#define JBLIB_CONTAINERS_ARRAY3D_FFTW_H

#include <fftw3.h>

#include <algorithm>
#include <iostream>
#include <new>
#include <string>

#ifdef BOUNDSCHECK
#include <stdexcept>
#include <sstream>
#endif

#include "jblib/containers/array/array_template.h"
#include "jblib/sys/intrinsic.h"

namespace jblib {
  template <typename Idx_>
  class Array <fftw_complex, 3, Idx_> {
   public:
    typedef fftw_complex        value_type;
    typedef fftw_complex&       reference;
    typedef const fftw_complex& const_reference;
    typedef fftw_complex*       pointer;
    typedef const fftw_complex* const_pointer;
    typedef Idx_                          size_type;

    Array();
    explicit Array(const size_type size0, const size_type size1,
      const size_type size2);
    explicit Array(const Array<fftw_complex, 3, Idx_>& other);
    Array(const size_type size0, const size_type size1,
      const size_type size2,
      const value_type initial_value);
    ~Array();

    Array<fftw_complex, 3, Idx_>& operator=(Array<fftw_complex, 3, Idx_> rhs);

          reference operator() (const size_type i, const size_type j,
            const size_type k);
    const_reference operator() (const size_type i, const size_type j,
      const size_type k) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

          pointer data();
    const_pointer data() const;

    const size_type& size(const size_type i) const;
    const size_type elements() const;

    void resize(const size_type size0, const size_type size1,
      const size_type size2);

    bool is_allocated() const;

    template<typename FTp_, typename FIdx_>
    friend void swap(Array<FTp_, 3, FIdx_>& first, Array<FTp_, 3, FIdx_>& second);

   private:
#ifdef BOUNDSCHECK
    bool is_range_valid(const size_type i) const;
    bool is_range_valid(const size_type i, const size_type j,
      const size_type k) const;
    std::string range_error_message(const size_type i) const;
    std::string range_error_message(const size_type i, const size_type j,
      const size_type k) const;
#endif

    size_type size0_;
    size_type size1_;
    size_type size2_;
    pointer   data_;
  };

//-----------------------------------------------------------------------------
// constructors
//-----------------------------------------------------------------------------

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>::
  Array()
  : size0_(0), size1_(0), size2_(0), data_(NULL)
  {}

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2)
  : size0_(size0), size1_(size1), size2_(size2),
    data_(size0_*size1_*size2_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*sizeof(fftw_complex))): NULL)
  {
  }

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>::
  Array(const Array<fftw_complex, 3, Idx_ >& other)
  : size0_(other.size0_), size1_(other.size1_), size2_(other.size2_),
    data_(size0_*size1_*size2_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*sizeof(fftw_complex))): NULL) {
      std::copy(other.data_, (other.data_ + elements()), data_);
  }

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2,
    const value_type initial_value)
    : size0_(size0), size1_(size1), size2_(size2),
    data_(size0_*size1_*size2_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*sizeof(data_))): NULL) {
    for (size_type i = 0, iend = elements(); i != iend; ++i) {
      data_[i] = initial_value;
    }
  }

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>::
  ~Array() {
    if (is_allocated()) {
      fftw_free(data_);
      data_ = NULL;
    }
  }

//-----------------------------------------------------------------------------
// operators
//-----------------------------------------------------------------------------

  template <typename Idx_>
  Array<fftw_complex, 3, Idx_>& Array<fftw_complex, 3, Idx_>::
  operator=(Array<fftw_complex, 3, Idx_> rhs) {
    using std::swap;
    swap(*this, rhs);
    return *this;
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::reference
  Array<fftw_complex, 3, Idx_>::
  operator() (const size_type i, const size_type j, const size_type k) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i, j, k)) {
      throw std::out_of_range(range_error_message(i, j, k));
    }
#endif
    return data_[((i*size1_+j)*size2_+k)];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::const_reference
  Array<fftw_complex, 3, Idx_>::
  operator() (const size_type i, const size_type j, const size_type k) const {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i, j, k)) {
      throw std::out_of_range(range_error_message(i, j, k));
    }
#endif
    return data_[((i*size1_+j)*size2_+k)];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::reference
  Array<fftw_complex, 3, Idx_>::
  operator[] (const size_type i) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::const_reference
  Array<fftw_complex, 3, Idx_>::
  operator[] (const size_type i) const {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

//-----------------------------------------------------------------------------
// member functions
//-----------------------------------------------------------------------------

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::pointer
  Array<fftw_complex, 3, Idx_>::
  data() {
    assert(is_allocated());
    return data_;
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 3, Idx_>::const_pointer
  Array<fftw_complex, 3, Idx_>::
  data() const {
    assert(is_allocated());
    return data_;
  }

  template <typename Idx_>
  inline const typename Array<fftw_complex, 3, Idx_>::size_type&
  Array<fftw_complex, 3, Idx_>::
  size(const size_type i) const {
    assert((i < 3) && !(i < 0));
    return (&size0_)[i];
  }

  template <typename Idx_>
  inline const typename Array<fftw_complex, 3, Idx_>::size_type
  Array<fftw_complex, 3, Idx_>::
  elements() const {
    return size0_*size1_*size2_;
  }

  template <typename Idx_>
  void
  Array<fftw_complex, 3, Idx_>::
  resize(const size_type new_size0, const size_type new_size1,
    const size_type new_size2) {
    using std::swap;
    Array< fftw_complex, 3, Idx_> newArray(new_size0, new_size1, new_size2);
      // copy the smaller array dimensions
    if (is_allocated()) {
      for (size_type i = 0, iend = std::min(new_size0, size0_); i != iend; ++i) {
        for (size_type j = 0, jend = std::min(new_size1, size1_); j != jend; ++j) {
          for (size_type k = 0, kend = std::min(new_size2, size2_); k != kend; ++k) {
            newArray(i, j, k)[0] = data_[((i*size1_+j)*size2_+k)][0];
            newArray(i, j, k)[1] = data_[((i*size1_+j)*size2_+k)][1];
          }
        }
      }
    }
    swap(*this, newArray);
  }

  template <typename Idx_>
  inline bool
  Array<fftw_complex, 3, Idx_>::
  is_allocated() const {
    if(data_ != NULL) { return true; }
    return false;
  }

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline bool
  Array<fftw_complex, 3, Idx_>::
  is_range_valid(const size_type i) const {
    return ((i < elements()) && !(i < 0) );
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline bool
  Array<fftw_complex, 3, Idx_>::
  is_range_valid(const size_type i, const size_type j, const size_type k) const {
    return (((i < size0_) && !(i < 0)) && ((j < size1_) && !(j < 0))
       && ((k < size2_) && !(k < 0)) );
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline std::string
  Array<fftw_complex, 3, Idx_>::
  range_error_message(const size_type i) const {
    std::ostringstream message;
    message << "Array<3>::operator[] ";
    message << "subscript: [ " << i <<  " ] ";
    message << "range: ( " << elements() << " ) ";
    return message.str();
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline std::string
  Array<fftw_complex, 3, Idx_>::
  range_error_message(const size_type i, const size_type j, const size_type k) const {
    std::ostringstream message;
    message << "Array<3>::operator() ";
    message << "subscript: ( " << i << " , " << j << " , " << k << " ) ";
    message << "range: ( " << size0_ << " , " << size1_ << " , " << size2_ << " ) ";
    return message.str();
  }
#endif

}  // namespace jblib

#endif  // JBLIB_CONTAINERS_ARRAY3D_FFTW_H
