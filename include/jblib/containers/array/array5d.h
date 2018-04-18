// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_ARRAY5D_H
#define JBLIB_CONTAINERS_ARRAY5D_H

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
  template <typename Tp_, typename Idx_>
  class Array <Tp_, 5, Idx_> {
   public:
    typedef Tp_                           value_type;
    typedef Tp_& restrict aligned64       reference;
    typedef const Tp_& restrict aligned64 const_reference;
    typedef Tp_* restrict aligned64       pointer;
    typedef const Tp_* restrict aligned64 const_pointer;
    typedef Idx_                          size_type;

    Array();

    Array(const size_type size0,
                   const size_type size1,
                   const size_type size2,
                   const size_type size3,
                   const size_type size4);

    Array(const Array<Tp_, 5, Idx_>& other);

    Array(const size_type size0,
          const size_type size1,
          const size_type size2,
          const size_type size4,
          const size_type size5,
          const value_type initial_value);

    ~Array();

    Array<Tp_, 5, Idx_>& operator=(Array<Tp_, 5, Idx_> rhs);

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

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>::
  Array()
  : size0_(0), size1_(0), size2_(0), size3_(0), size4_(0), data_(NULL)
  {}

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2,
    const size_type size3, const size_type size4)
  : size0_(size0), size1_(size1), size2_(size2), size3_(size3), size4_(size4),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (allocate_aligned64(size0_*size1_*size2_*size3_*size4_*sizeof(Tp_))): NULL)
  {}

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>::
  Array(const Array<Tp_, 5, Idx_ >& other)
  : size0_(other.size0_), size1_(other.size1_), size2_(other.size2_),
    size3_(other.size3_), size4_(other.size4_),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (allocate_aligned64(size0_*size1_*size2_*size3_*size4_*sizeof(Tp_))): NULL) {
      std::copy(other.data_, (other.data_ + elements()), data_);
  }

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>::
  Array(const size_type size0, const size_type size1, const size_type size2,
    const size_type size3, const size_type size4, const value_type initial_value)
    : size0_(size0), size1_(size1), size2_(size2), size3_(size3), size4_(size4),
    data_(size0_*size1_*size2_*size3_*size4_ ? reinterpret_cast<pointer>
      (allocate_aligned64(size0_*size1_*size2_*size3_*size4_*sizeof(Tp_))): NULL) {
    for (size_type i = 0, iend = elements(); i != iend; ++i) {
      data_[i] = initial_value;
    }
  }

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>::
  ~Array() {
    if (is_allocated()) {
      free(data_);
      data_ = NULL;
    }
  }

//-----------------------------------------------------------------------------
// operators
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  Array<Tp_, 5, Idx_>& Array<Tp_, 5, Idx_>::
  operator=(Array<Tp_, 5, Idx_> rhs) {
    using std::swap;
    swap(*this, rhs);
    return *this;
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::reference
  Array<Tp_, 5, Idx_>::
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

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::const_reference
  Array<Tp_, 5, Idx_>::
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

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::reference
  Array<Tp_, 5, Idx_>::
  operator[] (const size_type i) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::const_reference
  Array<Tp_, 5, Idx_>::
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

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::pointer
  Array<Tp_, 5, Idx_>::
  data() {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 5, Idx_>::const_pointer
  Array<Tp_, 5, Idx_>::
  data() const {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline const typename Array<Tp_, 5, Idx_>::size_type&
  Array<Tp_, 5, Idx_>::
  size(const size_type i) const {
    assert((i < 5) && !(i < 0));
    switch(i) {
      case 0 :
        return size0_;
      case 1 :
        return size1_;
      case 2 :
        return size2_;
      case 3 :
        return size3_;
      case 4 :
        return size4_;
    }
    throw std::out_of_range("size(i) must have i < 4");
  }

  template <typename Tp_, typename Idx_>
  inline const typename Array<Tp_, 5, Idx_>::size_type
  Array<Tp_, 5, Idx_>::
  elements() const {
    return size0_*size1_*size2_*size3_*size4_;
  }

  template <typename Tp_, typename Idx_>
  void
  Array<Tp_, 5, Idx_>::
  resize(const size_type new_size0, const size_type new_size1,
    const size_type new_size2, const size_type new_size3, const size_type new_size4) {
    using std::swap;

    // don't resize if already the same size
    if ((new_size0 == size0_) && (new_size1 == size1_) && (new_size2 == size2_) && (new_size3 == size3_) && (new_size4 == size4_)) {
      return;
    }

    Array< Tp_, 5, Idx_> newArray(new_size0, new_size1, new_size2, new_size3, new_size4);
      // copy the smaller array dimensions
    for (size_type i = 0, iend = std::min(new_size0, size0_); i != iend; ++i) {
      for (size_type j = 0, jend = std::min(new_size1, size1_); j != jend; ++j) {
        for (size_type k = 0, kend = std::min(new_size2, size2_); k != kend; ++k) {
          for (size_type l = 0, lend = std::min(new_size3, size3_); l != lend; ++l) {
            for (size_type m = 0, mend = std::min(new_size4, size4_); m != mend; ++m) {
              newArray(i, j, k, l, m) = data_[(((i*size1_+j)*size2_+k)*size3_+l)*size4_+m];
            }
          }
        }
      }
    }
    swap(*this, newArray);
  }

  template <typename Tp_, typename Idx_>
  void
  Array<Tp_, 5, Idx_>::
  zero() {
     std::fill(data_, data_ + (size0_ * size1_ * size2_ * size3_ * size4_), Tp_(0));
  }

  template <typename Tp_, typename Idx_>
  inline bool
  Array<Tp_, 5, Idx_>::
  is_allocated() const {
    return data_;
  }

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline bool
  Array<Tp_, 5, Idx_>::
  is_range_valid(const size_type i) const {
    return ((i < elements()) && !(i < 0) );
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline bool
  Array<Tp_, 5, Idx_>::
  is_range_valid(const size_type i, const size_type j, const size_type k,
    const size_type l, const size_type m) const {
    return (((i < size0_) && !(i < 0)) && ((j < size1_) && !(j < 0))
       && ((k < size2_) && !(k < 0)) && ((l < size3_) && !(l < 0))
       && ((m < size4_) && !(m < 0)));
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline std::string
  Array<Tp_, 5, Idx_>::
  range_error_message(const size_type i) const {
    std::ostringstream message;
    message << "Array<2>::operator[] ";
    message << "subscript: [ " << i <<  " ] ";
    message << "range: ( " << elements() << " ) ";
    return message.str();
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline std::string
  Array<Tp_, 5, Idx_>::
  range_error_message(const size_type i, const size_type j, const size_type k, const size_type l, const size_type m) const {
    std::ostringstream message;
    message << "Array<2>::operator() ";
    message << "subscript: ( " << i << " , " << j << " , " << k << " , " << l << " , " << m << " ) ";
    message << "range: ( " << size0_ << " , " << size1_ << " , " << size2_ << " , " << size3_ << " , " << size4_ << " ) ";
    return message.str();
  }
#endif

//-----------------------------------------------------------------------------
// friends
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  void swap(Array<Tp_, 5, Idx_>& first, Array<Tp_, 5, Idx_>& second) {
    using std::swap;
    swap(first.size0_, second.size0_);
    swap(first.size1_, second.size1_);
    swap(first.size2_, second.size2_);
    swap(first.size3_, second.size3_);
    swap(first.size4_, second.size4_);
    swap(first.data_, second.data_);
  }

}  // namespace jblib

#endif  // JBLIB_CONTAINERS_ARRAY5D_H
