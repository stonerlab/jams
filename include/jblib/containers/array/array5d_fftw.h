// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_ARRAY5D_FFTW_H
#define JBLIB_CONTAINERS_ARRAY5D_FFTW_H

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
  class Array <fftw_complex, 5, Idx_> {
   public:
    typedef fftw_complex         value_type;
    typedef fftw_complex&        reference;
    typedef const fftw_complex&  const_reference;
    typedef fftw_complex*        pointer;
    typedef const fftw_complex*  const_pointer;
    typedef Idx_                 size_type;

    Array();

    explicit Array(const size_type size0,
                   const size_type size1,
                   const size_type size2,
                   const size_type size3,
                   const size_type size4);

    Array(const Array<fftw_complex, 5, Idx_>& other);

    Array(const size_type size0,
          const size_type size1,
          const size_type size2,
          const size_type size4,
          const size_type size5,
          const value_type initial_value);

    ~Array();

    Array<fftw_complex, 5, Idx_>& operator=(Array<fftw_complex, 5, Idx_> rhs);

          reference operator() (const size_type i, const size_type j,
            const size_type k, const size_type l, const size_type m);
    const_reference operator() (const size_type i, const size_type j,
            const size_type k, const size_type l, const size_type m) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

          pointer data();
    const_pointer data() const;

    const size_type& size(const size_type i) const;
    const size_type elements() const;

    void resize(const size_type size0, const size_type size1,
      const size_type size2, const size_type size3, const size_type size4);
    void zero();


    bool is_allocated() const;

    template<typename FTp_, typename FIdx_>
    friend void swap(Array<FTp_, 5, FIdx_>& first, Array<FTp_, 5, FIdx_>& second);
   private:
#ifdef BOUNDSCHECK
    bool is_range_valid(const size_type i) const;
    bool is_range_valid(const size_type i, const size_type j,
      const size_type k, const size_type l, const size_type m) const;
    std::string range_error_message(const size_type i) const;
    std::string range_error_message(const size_type i, const size_type j,
      const size_type k, const size_type l, const size_type m) const;
#endif

    size_type size0_;
    size_type size1_;
    size_type size2_;
    size_type size3_;
    size_type size4_;
    pointer   data_;
  };

//-----------------------------------------------------------------------------
// constructors
//-----------------------------------------------------------------------------

  template <typename Idx_>
  Array<fftw_complex, 5, Idx_>::
  Array()
  : size0_(0), size1_(0), size2_(0), size3_(0), size4_(0), data_(NULL)
  {}

  template <typename Idx_>
  Array<fftw_complex, 5, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2,
    const size_type size3, const size_type size4)
  : size0_(size0), size1_(size1), size2_(size2), size3_(size3), size4_(size4),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*size3_*size4_*sizeof(fftw_complex))): NULL)
  {}

  template <typename Idx_>
  Array<fftw_complex, 5, Idx_>::
  Array(const Array<fftw_complex, 5, Idx_ >& other)
  : size0_(other.size0_), size1_(other.size1_), size2_(other.size2_),
    size3_(other.size3_), size4_(other.size4_),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*size3_*size4_*sizeof(fftw_complex))): NULL) {
      std::memcpy(data_, other.data_, elements() * sizeof(fftw_complex));
  }

  template <typename Idx_>
  Array<fftw_complex, 5, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2,
    const size_type size3, const size_type size4, const value_type initial_value)
    : size0_(size0), size1_(size1), size2_(size2), size3_(size3), size4_(size4),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (fftw_malloc(size0_*size1_*size2_*size3_*size4_*sizeof(data_))): NULL) {
    for (size_type i = 0, iend = elements(); i != iend; ++i) {
      data_[i] = initial_value;
    }
  }

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

  template <typename Idx_>
  Array<fftw_complex, 5, Idx_>::
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
  Array<fftw_complex, 5, Idx_>& Array<fftw_complex, 5, Idx_>::
  operator=(Array<fftw_complex, 5, Idx_> rhs) {
    using std::swap;
    swap(*this, rhs);
    return *this;
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 5, Idx_>::reference
  Array<fftw_complex, 5, Idx_>::
  operator() (const size_type i, const size_type j, const size_type k,
    const size_type l, const size_type m) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i, j, k, l, m)) {
      throw std::out_of_range(range_error_message(i, j, k, l, m));
    }
#endif
    return data_[(((i*size1_+j)*size2_+k)*size3_+l)*size4_+m];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 5, Idx_>::const_reference
  Array<fftw_complex, 5, Idx_>::
  operator() (const size_type i, const size_type j, const size_type k,
    const size_type l, const size_type m) const {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i, j, k, l, m)) {
      throw std::out_of_range(range_error_message(i, j, k, l, m));
    }
#endif
    return data_[(((i*size1_+j)*size2_+k)*size3_+l)*size4_+m];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 5, Idx_>::reference
  Array<fftw_complex, 5, Idx_>::
  operator[] (const size_type i) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 5, Idx_>::const_reference
  Array<fftw_complex, 5, Idx_>::
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
  inline typename Array<fftw_complex, 5, Idx_>::pointer
  Array<fftw_complex, 5, Idx_>::
  data() {
    assert(is_allocated());
    return data_;
  }

  template <typename Idx_>
  inline typename Array<fftw_complex, 5, Idx_>::const_pointer
  Array<fftw_complex, 5, Idx_>::
  data() const {
    assert(is_allocated());
    return data_;
  }

  template <typename Idx_>
  inline const typename Array<fftw_complex, 5, Idx_>::size_type&
  Array<fftw_complex, 5, Idx_>::
  size(const size_type i) const {
    assert((i < 5) && !(i < 0));
    return (&size0_)[i];
  }

  template <typename Idx_>
  inline const typename Array<fftw_complex, 5, Idx_>::size_type
  Array<fftw_complex, 5, Idx_>::
  elements() const {
    return size0_*size1_*size2_*size3_*size4_;
  }

  template <typename Idx_>
  void
  Array<fftw_complex, 5, Idx_>::
  resize(const size_type new_size0, const size_type new_size1,
    const size_type new_size2, const size_type new_size3, const size_type new_size4) {
    using std::swap;
    Array< fftw_complex, 5, Idx_> newArray(new_size0, new_size1, new_size2, new_size3, new_size4);
      // copy the smaller array dimensions
    if ( is_allocated() ){
      for (size_type i = 0, iend = std::min(new_size0, size0_); i != iend; ++i) {
        for (size_type j = 0, jend = std::min(new_size1, size1_); j != jend; ++j) {
          for (size_type k = 0, kend = std::min(new_size2, size2_); k != kend; ++k) {
            for (size_type l = 0, lend = std::min(new_size3, size3_); l != lend; ++l) {
              for (size_type m = 0, mend = std::min(new_size4, size4_); m != mend; ++m) {
                newArray(i, j, k, l, m)[0] = data_[(((i*size1_+j)*size2_+k)*size3_+l)*size4_+m][0];
                newArray(i, j, k, l, m)[1] = data_[(((i*size1_+j)*size2_+k)*size3_+l)*size4_+m][1];
              }
            }
          }
        }
      }
    }
    swap(*this, newArray);
  }

  template <typename Idx_>
  void
  Array<fftw_complex, 5, Idx_>::
  zero() {
    for (size_type i = 0; i < elements(); ++i) {
      data_[i][0] = 0.0; data_[i][1] = 0.0;
    }
  }

  template <typename Idx_>
  inline bool
  Array<fftw_complex, 5, Idx_>::
  is_allocated() const {
    if(data_ != NULL) { return true; }
    return false;
  }

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline bool
  Array<fftw_complex, 5, Idx_>::
  is_range_valid(const size_type i) const {
    return ((i < elements()) && !(i < 0) );
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline bool
  Array<fftw_complex, 5, Idx_>::
  is_range_valid(const size_type i, const size_type j, const size_type k,
    const size_type l, const size_type m) const {
    return (((i < size0_) && !(i < 0)) && ((j < size1_) && !(j < 0))
       && ((k < size2_) && !(k < 0)) && ((l < size3_) && !(l < 0))
       && ((m < size4_) && !(m < 0)));
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline std::string
  Array<fftw_complex, 5, Idx_>::
  range_error_message(const size_type i) const {
    std::ostringstream message;
    message << "Array<5>::operator[] ";
    message << "subscript: [ " << i <<  " ] ";
    message << "range: ( " << elements() << " ) ";
    return message.str();
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Idx_>
  inline std::string
  Array<fftw_complex, 5, Idx_>::
  range_error_message(const size_type i, const size_type j, const size_type k, const size_type l, const size_type m) const {
    std::ostringstream message;
    message << "Array<5>::operator() ";
    message << "subscript: ( " << i << " , " << j << " , " << k << " , " << l << " , " << m << " ) ";
    message << "range: ( " << size0_ << " , " << size1_ << " , " << size2_ << " , " << size3_ << " , " << size4_ << " ) ";
    return message.str();
  }
#endif

}  // namespace jblib

#endif  // JBLIB_CONTAINERS_ARRAY5D_FFTW_H
