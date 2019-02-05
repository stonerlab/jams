// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_CUDA_ARRAY1D_H
#define JBLIB_CONTAINERS_CUDA_ARRAY1D_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <new>
#include <iostream>
#include <stdexcept>

#include "jblib/containers/cuda_array/cuda_array_template.h"
#include "jblib/containers/array.h"

namespace jblib {
  template <typename Tp_, typename Idx_>
  class CudaArray<Tp_, 1, Idx_> {
   public:
    typedef Tp_          value_type;
    typedef Tp_&         reference;
    typedef const Tp_&   const_reference;
    typedef Tp_*         pointer;
    typedef const Tp_*   const_pointer;
    typedef Idx_         size_type;

    explicit CudaArray(const Idx_ size0 = 0);
    CudaArray(const CudaArray<Tp_, 1, Idx_>& other);
    CudaArray(const Idx_ size0, const Tp_ initial_value);

    ~CudaArray();

    CudaArray<Tp_, 1, Idx_>& operator=(CudaArray<Tp_, 1, Idx_> rhs);

    // Construct a CudaArray from a host Array

    template <int HostDim_>
    CudaArray(const Array<Tp_, HostDim_, Idx_>& host_array);

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
    void zero(cudaStream_t &stream);

    bool is_allocated() const;

    template <int HostDim_>
    void copy_from_host_array(const Array<Tp_, HostDim_, Idx_> &host_array);

        template <int HostDim_>
    void copy_to_host_array(Array<Tp_, HostDim_, Idx_> &host_array);


    friend void swap(CudaArray<Tp_, 1, Idx_>& first, CudaArray<Tp_, 1, Idx_>& second) {
      std::swap(first.size_, second.size_);
      std::swap(first.data_, second.data_);
    }

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
  CudaArray<Tp_, 1, Idx_>::
  CudaArray(const size_type size0)
  : size_(size0), data_(NULL) {
    if (size_ != 0) {
      if (cudaMalloc(reinterpret_cast<void**>(&data_), size_*sizeof(Tp_))
        != cudaSuccess) {
        throw std::bad_alloc();
      }
    }
  }

  template <typename Tp_, typename Idx_>
  CudaArray<Tp_, 1, Idx_>::
  CudaArray(const CudaArray<Tp_, 1, Idx_>& other) {
    size_ = other.size_;
    if (size_ != 0) {
      cudaMalloc(reinterpret_cast<void**>(&data_), size_*sizeof(Tp_));
      cudaMemcpy(data_, other.data_, (size_t)(size_*sizeof(Tp_)), cudaMemcpyDeviceToDevice);
    }
  }

  template <typename Tp_, typename Idx_>
  CudaArray<Tp_, 1, Idx_>::
  CudaArray(const size_type size0, const Tp_ initial_value)
  : CudaArray(Array<Tp_, 1, Idx_>(size0, initial_value)) {};

  template <typename Tp_, typename Idx_>
  template<int HostDim_>
  CudaArray<Tp_, 1, Idx_>::
  CudaArray(const Array<Tp_, HostDim_, Idx_>& host_array) {
    size_ = host_array.elements();
    if (size_ != 0) {
      if (cudaMalloc(reinterpret_cast<void**>(&data_), size_*sizeof(Tp_))
        != cudaSuccess) {
        throw std::bad_alloc();
      }
    }
    copy_from_host_array(host_array);
  }

//-----------------------------------------------------------------------------
// destructor
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  CudaArray<Tp_, 1, Idx_>::
  ~CudaArray() {
    cudaFree(data_);
  }

//-----------------------------------------------------------------------------
// operators
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  CudaArray<Tp_, 1, Idx_>& CudaArray<Tp_, 1, Idx_>::
  operator=(CudaArray<Tp_, 1, Idx_> rhs) {
    swap(*this, rhs);
    return *this;
  }

//-----------------------------------------------------------------------------
// member functions
//-----------------------------------------------------------------------------

  template <typename Tp_, typename Idx_>
  inline typename CudaArray<Tp_, 1, Idx_>::pointer
  CudaArray<Tp_, 1, Idx_>::
  data() {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline typename CudaArray<Tp_, 1, Idx_>::const_pointer
  CudaArray<Tp_, 1, Idx_>::
  data() const {
    assert(is_allocated());
    return data_;
  }

  template <typename Tp_, typename Idx_>
  inline const typename CudaArray<Tp_, 1, Idx_>::size_type&
  CudaArray<Tp_, 1, Idx_>::
  size() const {
    return size_;
  }

  template <typename Tp_, typename Idx_>
  inline const typename CudaArray<Tp_, 1, Idx_>::size_type&
  CudaArray<Tp_, 1, Idx_>::
  elements() const {
    return size_;
  }

  template <typename Tp_, typename Idx_>
  void CudaArray<Tp_, 1, Idx_>::resize(const size_type size0){
    CudaArray< Tp_, 1, Idx_> new_cuda_array(size0);
    // copy the smaller array dimensions
    if (size_ != 0) {
      if (cudaMemcpy(new_cuda_array.data_, data_,
        (size_t)(std::min(size0, size_)*sizeof(Tp_)),
        cudaMemcpyDeviceToDevice) != cudaSuccess) {
        throw std::runtime_error("resize fail - cudaMemcpy failure.");
      }
    }
    swap(*this, new_cuda_array);
  }

  template <typename Tp_, typename Idx_>
  void CudaArray<Tp_, 1, Idx_>::zero(){
    if (size_ != 0) {
#ifdef NDEBUG
      cudaMemset(data_, 0, size_*sizeof(Tp_));
#else
      if (cudaMemset(data_, 0, size_*sizeof(Tp_)) != cudaSuccess) {
        throw std::runtime_error("zero fail - cudaMemset failure.");
      }
#endif
    }
  }

  template <typename Tp_, typename Idx_>
  void CudaArray<Tp_, 1, Idx_>::zero(cudaStream_t &stream){
    if (size_ != 0) {
#ifdef NDEBUG
      cudaMemsetAsync(data_, 0, size_*sizeof(Tp_), stream);
#else
      if (cudaMemsetAsync(data_, 0, size_*sizeof(Tp_), stream) != cudaSuccess) {
        throw std::runtime_error("zero fail - cudaMemset failure.");
      }
#endif
    }
  }

  template <typename Tp_, typename Idx_>
  inline bool
  CudaArray<Tp_, 1, Idx_>::
  is_allocated() const {
    return data_;
  }

  template <typename Tp_, typename Idx_>
    template <int HostDim_>
  void CudaArray<Tp_, 1, Idx_>::
  copy_from_host_array(const Array<Tp_, HostDim_, Idx_> &host_array) {
    if (size_ != host_array.elements()) {
      throw std::runtime_error("copyFromHostArray fail - size mismatch");
    }

    if (cudaMemcpy(data_, host_array.data(), (size_t)(size_*sizeof(Tp_)),
      cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("copyFromHostArray fail - cudaMemcpy failure.");
    }
  }

  template <typename Tp_, typename Idx_>
    template <int HostDim_>
  void CudaArray<Tp_, 1, Idx_>::
  copy_to_host_array(Array<Tp_, HostDim_, Idx_> &host_array) {
    if (size_ != host_array.elements()) {
      throw std::runtime_error("copyToHostArray fail - size mismatch");
    }
    if (cudaMemcpy(host_array.data(),data_,(size_t)(size_*sizeof(Tp_)),cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("copyToHostArray fail - cudaMemcpy failure.");
    }
  }

}  // namespace jblib
#endif  // JBLIB_CONTAINERS_CUDA_ARRAY1D_H
