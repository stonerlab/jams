// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_ARRAY1D_H
#define JBLIB_CONTAINERS_ARRAY1D_H

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
  class Array <Tp_, 1, Idx_> {
   public:
    typedef Tp_                           value_type;
    typedef Tp_& restrict aligned64       reference;
    typedef const Tp_& restrict aligned64 const_reference;
    typedef Tp_* restrict aligned64       pointer;
    typedef const Tp_* restrict aligned64 const_pointer;
    typedef Idx_                          size_type;

    Array();
    Array(const size_type size0);
    Array(const Array<Tp_, 1, Idx_>& other);
    Array(const size_type size0, const value_type initial_value);
    ~Array();

    Array<Tp_, 1, Idx_>& operator=(Array<Tp_, 1, Idx_> rhs);

          reference operator() (const size_type i);
    const_reference operator() (const size_type i) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

          pointer data();
    const_pointer data() const;

    const size_type& size() const;
    const size_type& elements() const;

    void resize(const size_type size0);
    void zero();

    bool is_allocated() const;

    template<typename FTp_, typename FIdx_>
    friend void swap(Array<FTp_, 1, FIdx_>& first, Array<FTp_, 1, FIdx_>& second);

   private:
#ifdef BOUNDSCHECK
    bool is_range_valid(const Idx_ i) const;
    std::string range_error_message(const Idx_ i) const;
#endif

    size_type size_;
    pointer   data_;
  };

//-----------------------------------------------------------------------------
// constructors
//-----------------------------------------------------------------------------
  template <typename Tp_, typename Idx_>
  Array<Tp_, 1, Idx_>::
  Array()
  : size_(0), data_(NULL)
  {}

  template <typename Tp_, typename Idx_>
  Array<Tp_, 1, Idx_>::
  Array(const Idx_ size0)
  : size_(size0),
    data_(size_ ? reinterpret_cast<pointer>
      (allocate_aligned64(size_*sizeof(Tp_))): NULL)
  {}

  template <typename Tp_, typename Idx_>
  Array<Tp_, 1, Idx_>::
  Array(const Array<Tp_, 1, Idx_ >& other)
  : size_(other.size_),
  data_(size_ ? reinterpret_cast<pointer>
    (allocate_aligned64(size_*sizeof(Tp_))): NULL ) {
    std::copy(other.data_, (other.data_ + size_), data_);
  }

  template <typename Tp_, typename Idx_>
  Array<Tp_, 1, Idx_>::
  Array(const Idx_ size0, const Tp_ initial_value)
    : size_(size0),
      data_(size_ ? reinterpret_cast<pointer>
        (allocate_aligned64(size_*sizeof(Tp_))): NULL) {
    for (size_type i = 0; i != size_; ++i) {
      data_[i] = initial_value;
    }
  }

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  Array<Tp_, 1, Idx_>::
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
  Array<Tp_, 1, Idx_>& Array<Tp_, 1, Idx_>::
  operator=(Array<Tp_, 1, Idx_> rhs) {
    using std::swap;
    swap(*this, rhs);
    return *this;
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 1, Idx_>::reference
  Array<Tp_, 1, Idx_>::
  operator() (const Idx_ i) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 1, Idx_>::const_reference
  Array<Tp_, 1, Idx_>::
  operator() (const Idx_ i) const {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 1, Idx_>::reference
  Array<Tp_, 1, Idx_>::
  operator[] (const Idx_ i) {
    assert(is_allocated());
#ifdef BOUNDSCHECK
    if (!is_range_valid(i)) {throw std::out_of_range(range_error_message(i));}
#endif
    return data_[i];
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 1, Idx_>::const_reference
  Array<Tp_, 1, Idx_>::
  operator[] (const Idx_ i) const {
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
  inline typename Array<Tp_, 1, Idx_>::pointer
  Array<Tp_, 1, Idx_>::
  data() {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline typename Array<Tp_, 1, Idx_>::const_pointer
  Array<Tp_, 1, Idx_>::
  data() const {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline const typename Array<Tp_, 1, Idx_>::size_type&
  Array<Tp_, 1, Idx_>::
  size() const {
    return size_;
  }

  template <typename Tp_, typename Idx_>
  inline const typename Array<Tp_, 1, Idx_>::size_type&
  Array<Tp_, 1, Idx_>::
  elements() const {
    return size_;
  }

  template <typename Tp_, typename Idx_>
  void
  Array<Tp_, 1, Idx_>::
  resize(const size_type size0) {
    using std::swap;

    // don't resize if already the same size
    if ((size0 == size_)) {
      return;
    }

    Array< Tp_, 1, Idx_> newArray(size0);
      // copy the smaller array dimensions
    for (size_type i = 0, iend = std::min(size0, size_); i != iend; ++i) {
      newArray(i) = data_[i];
    }
    swap(*this, newArray);
  }

  template <typename Tp_, typename Idx_>
  void
  Array<Tp_, 1, Idx_>::
  zero() {
    // std::fill(data_, data_ + (size0_ * size1_), Tp_(0));
    memset(data_, 0.0, (size_)*sizeof(Tp_));
  }

  template <typename Tp_, typename Idx_>
  inline bool
  Array<Tp_, 1, Idx_>::
  is_allocated() const {
    return data_;
  }

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline bool
  Array<Tp_, 1, Idx_>::
  is_range_valid(const Idx_ i) const {
    return (i < size_);
  }
#endif

#ifdef BOUNDSCHECK
  template <typename Tp_, typename Idx_>
  inline std::string
  Array<Tp_, 1, Idx_>::
  range_error_message(const Idx_ i) const {
    std::ostringstream message;
    message << "Array<1>::operator() ";
    message << "subscript: ( " << i <<  " ) ";
    message << "range: ( " << size_ << " ) ";
    return message.str();
  }
#endif

//-----------------------------------------------------------------------------
// friends
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  inline void swap(Array<Tp_, 1, Idx_>& first, Array<Tp_, 1, Idx_>& second) {
    using std::swap;
    std::swap(first.size_, second.size_);
    std::swap(first.data_, second.data_);
  }
}  // namespace jblib

#endif  // JBLIB_CONTAINERS_ARRAY1D_H
