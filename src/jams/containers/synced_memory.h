//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_SYNCED_MEMORY_H
#define JAMS_SYNCED_MEMORY_H

#include <limits>
#include <iostream>

#include "jams/helpers/utils.h"

#if HAS_CUDA
#include <cuda_runtime.h>
#include "jams/cuda/cuda_common.h"
#endif


// toggle printing all host/device synchronization calls to cout
#define SYNCEDMEMORY_PRINT_MEMCPY 0

// toggle printing all host/device memset calls to cout
#define SYNCEDMEMORY_PRINT_MEMSET 0

// toggles support for using SyncedMemory in the global namespace
#define SYNCEDMEMORY_ALLOW_GLOBAL 1
//
// If SyncedMemory is used in a global context then free()
// calls CUDA routines after the CUDA context has been unloaded.
// The calls then return cudaErrorCudartUnloading as the status.
// This flag avoids checking the return status in free().

// toggle checking free memory on device before allocating
#define SYNCEDMEMORY_CHECK_FREE_MEMORY 1

// toggle zeroing of host/device memory immediately after allocation
#define SYNCEDMEMORY_ZERO_ON_ALLOCATION 0

// memory alignment for host memory (if supported)
#define SYNCEDMEMORY_HOST_ALIGNMENT 64

namespace jams {

template <class T>
class SyncedMemory {
public:
    template<class F>
    friend void swap(SyncedMemory<F>& lhs, SyncedMemory<F>& rhs);

    using value_type      = T;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using size_type       = std::size_t;

    enum class SyncStatus {
        UNINITIALIZED, SYNCHRONIZED, DEVICE_IS_MUTATED, HOST_IS_MUTATED };

    SyncedMemory() = default;

    ~SyncedMemory() {
      free_host_memory();
      free_device_memory();
    }

    // disallow copy constructor
    SyncedMemory(const SyncedMemory&) = delete;

    // disallow assignment operator
    SyncedMemory& operator=(const SyncedMemory&) = delete;

    // construct for a given size
    inline explicit SyncedMemory(size_type size) : size_(size) {}

    // construct for a given size and initial value
    inline SyncedMemory(size_type size, const T& x) : size_(size) {
      std::fill(mutable_host_data(), mutable_host_data() + size_, x);
    }

    // get size of data
    inline constexpr size_type size() const noexcept { return size_; }

    // get size of memory in bytes
    inline constexpr size_type memory() const noexcept { return size_ * sizeof(value_type); }

    // get maximum theoretical size of data
    inline constexpr size_type max_size() const noexcept;

    // accessors
    inline const_pointer const_host_data();
    inline const_pointer const_device_data();

    inline pointer mutable_host_data();
    inline pointer mutable_device_data();

    // modifiers
    inline void clear() { resize(0); }

    // zero all elements of the data
    inline void zero();

    // resize the data (destructive, reallocates)
    inline void resize(size_type new_size);

    // copy another SyncedMemory including state and only the GPU or CPU allocated memory
    inline void copy_from(const SyncedMemory&);

private:
    // copy host data to the device
    void copy_to_device();

    // copy device data to the host
    void copy_to_host();

    // allocate host data with size number of elements
    void allocate_host_memory(size_type size);

    // allocate device data with size number of elements
    void allocate_device_memory(size_type size);

    // set device data to zero
    inline void zero_device();

    // set host data to zero
    inline void zero_host();

    inline constexpr size_type max_size_host() const noexcept;
    inline size_type max_size_device() const;
    inline size_type available_size_device() const;

    inline void free_host_memory();
    inline void free_device_memory();

    SyncStatus sync_status_ = SyncStatus::UNINITIALIZED;
    size_type size_         = 0;
    pointer host_ptr_       = nullptr;
    pointer device_ptr_     = nullptr;
};

template<class T>
void SyncedMemory<T>::allocate_device_memory(const SyncedMemory::size_type size) {
  #if HAS_CUDA
  if (size == 0) return;

  if (size > max_size_device()) {
    throw std::bad_alloc();
  }

  #if SYNCEDMEMORY_CHECK_FREE_MEMORY
  if (size > available_size_device()) {
    throw std::bad_alloc();
  }
  #endif

  assert(!device_ptr_);
  if (cudaMalloc(reinterpret_cast<void**>(&device_ptr_), size_ * sizeof(T)) != cudaSuccess) {
    throw std::bad_alloc();
  }
  assert(device_ptr_);
  #endif
}

template<class T>
void SyncedMemory<T>::allocate_host_memory(const SyncedMemory::size_type size) {
  if (size == 0) return;

  assert(!host_ptr_);
  #if HAS_CUDA
  if (cudaMallocHost(reinterpret_cast<void**>(&host_ptr_), size_ * sizeof(T)) != cudaSuccess) {
    throw std::bad_alloc();
  }
  #else
  if (posix_memalign(reinterpret_cast<void**>(&host_ptr_), SYNCEDMEMORY_HOST_ALIGNMENT, size * sizeof(T) ) != 0) {
    throw std::bad_alloc();
  }
  #endif
  assert(host_ptr_);
}

template<class T>
__attribute__((hot))
inline typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_host_data() {
  if (unlikely(sync_status_ == SyncStatus::UNINITIALIZED || sync_status_ == SyncStatus::DEVICE_IS_MUTATED)) {
    copy_to_host();
  }
  return host_ptr_;
}

template<class T>
__attribute__((hot))
inline typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_device_data() {
  if (unlikely(sync_status_ == SyncStatus::UNINITIALIZED || sync_status_ == SyncStatus::HOST_IS_MUTATED)) {
    copy_to_device();
  }
  return device_ptr_;
}

template<class T>
__attribute__((hot))
inline typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_host_data() {
  if (unlikely(sync_status_ == SyncStatus::UNINITIALIZED || sync_status_ == SyncStatus::DEVICE_IS_MUTATED)) {
    copy_to_host();
  }
  sync_status_ = SyncStatus::HOST_IS_MUTATED;
  return host_ptr_;
}

template<class T>
__attribute__((hot))
inline typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_device_data() {
  if (unlikely(sync_status_ == SyncStatus::UNINITIALIZED || sync_status_ == SyncStatus::HOST_IS_MUTATED)) {
    copy_to_device();
  }
  sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
  return device_ptr_;
}

template<class T>
void SyncedMemory<T>::copy_to_device() {
  #if HAS_CUDA
  switch(sync_status_) {
    case SyncStatus::UNINITIALIZED:
      allocate_device_memory(size_);
      #ifdef SYNCEDMEMORY_ZERO_ON_ALLOCATION
      zero_device();
      #endif
      sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
      break;
    case SyncStatus::HOST_IS_MUTATED:
      if (!device_ptr_ ) allocate_device_memory(size_);
      #if SYNCEDMEMORY_PRINT_MEMCPY
        std::cout << "INFO(SyncedMemory): cudaMemcpyHostToDevice" << std::endl;
      #endif
      assert(device_ptr_ && host_ptr_);
      CHECK_CUDA_STATUS(cudaMemcpy(device_ptr_, host_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice));
      sync_status_ = SyncStatus::SYNCHRONIZED;
      break;
    case SyncStatus::DEVICE_IS_MUTATED:
    case SyncStatus::SYNCHRONIZED:
      break;
  }
  #endif
}

template<class T>
void SyncedMemory<T>::copy_to_host() {
  switch(sync_status_) {
  case SyncStatus::UNINITIALIZED:
    allocate_host_memory(size_);
    #ifdef SYNCEDMEMORY_ZERO_ON_ALLOCATION
    zero_host();
    #endif
    sync_status_ = SyncStatus::HOST_IS_MUTATED;
    break;
  case SyncStatus::DEVICE_IS_MUTATED:
    #if HAS_CUDA
    if (!host_ptr_) allocate_host_memory(size_);
    #if SYNCEDMEMORY_PRINT_MEMCPY
      std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToHost" << std::endl;
    #endif
    assert(device_ptr_ && host_ptr_);
    CHECK_CUDA_STATUS(cudaMemcpy(host_ptr_, device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    sync_status_ = SyncStatus::SYNCHRONIZED;
    break;
    #endif
  case SyncStatus::HOST_IS_MUTATED:
  case SyncStatus::SYNCHRONIZED:
    break;
  }
}

template<class T>
inline void SyncedMemory<T>::zero_device() {
  if (size_ == 0) return;

  #if HAS_CUDA
  #if SYNCEDMEMORY_PRINT_MEMSET
    std::cout << "INFO(SyncedMemory): device zero" << std::endl;
  #endif
  assert(device_ptr_);
  CHECK_CUDA_STATUS(cudaMemset(device_ptr_, 0, size_ * sizeof(T)));
  #endif
}

template<class T>
inline void SyncedMemory<T>::zero_host() {
  if (size_ == 0) return;

  #if SYNCEDMEMORY_PRINT_MEMSET
    std::cout << "INFO(SyncedMemory): host zero" << std::endl;
  #endif
  assert(host_ptr_);
  memset(host_ptr_, 0, size_ * sizeof(T));
}

template<class T>
void SyncedMemory<T>::zero() {
  if (host_ptr_) {
    zero_host();
    sync_status_ = SyncStatus::HOST_IS_MUTATED;
  }
  #if HAS_CUDA
  if (device_ptr_) {
    zero_device();
    sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
  }
  if (host_ptr_ && device_ptr_) {
    sync_status_ = SyncStatus::SYNCHRONIZED;
  }
  #endif
}

template<class T>
void SyncedMemory<T>::free_host_memory() {
  if (host_ptr_) {
    #if HAS_CUDA
      auto status = cudaFreeHost(host_ptr_);
      #if SYNCEDMEMORY_ALLOW_GLOBAL
        assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
      #else
        assert(status == cudaSuccess)
      #endif
    #else
    free(host_ptr_);
    #endif
  }
  host_ptr_ = nullptr;
}

template<class T>
void SyncedMemory<T>::free_device_memory() {
  #if HAS_CUDA
  if (device_ptr_) {
    auto status = cudaFree(device_ptr_);
    #if SYNCEDMEMORY_ALLOW_GLOBAL
    assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
    #else
    assert(status == cudaSuccess)
    #endif
  }
  #endif
  device_ptr_ = nullptr;
}

template<class T>
constexpr typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size() const noexcept {
  #if HAS_CUDA
  return std::min(max_size_host(), max_size_device());
  #else
  return max_size_host();
  #endif
}

template<class T>
void SyncedMemory<T>::resize(SyncedMemory::size_type new_size) {
  if (size_ == new_size) return;

  size_ = new_size;
  free_host_memory();
  free_device_memory();
  sync_status_ = SyncStatus::UNINITIALIZED;
}

// max_size is for returning the maximum theoretical size a container can be,
// i.e. it is still possible that we cannot allocate this amount of memory.
// For CUDA we are interpreting this as the maximum x-dimension of a grid of
// thread blocks
template<class T>
typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size_device() const {
  #if HAS_CUDA
  int dev = 0;
  CHECK_CUDA_STATUS(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, dev));
  return prop.maxGridSize[0];
  #else
  return 0;
  #endif
}

template<class T>
typename SyncedMemory<T>::size_type SyncedMemory<T>::available_size_device() const {
  #if HAS_CUDA
  size_t free;
  size_t total;
  CHECK_CUDA_STATUS(cudaMemGetInfo(&free, &total));
  return free / sizeof(value_type);
  #else
  return 0;
  #endif
}

template<class T>
constexpr typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size_host() const noexcept {
  return std::numeric_limits<size_type>::max();
}

template<class T>
void swap(SyncedMemory<T>& lhs, SyncedMemory<T>& rhs) {
  using std::swap;
  swap(lhs.sync_status_, rhs.sync_status_);
  swap(lhs.size_, rhs.size_);
  swap(lhs.host_ptr_, rhs.host_ptr_);
  swap(lhs.device_ptr_, rhs.device_ptr_);
}

// creates a true copy of another SyncedMemory by only copying the allocated memory spaces
template<class T>
void SyncedMemory<T>::copy_from(const SyncedMemory<T>& other) {
  if (this == &other) return;

  if (size_ != other.size_) {
    resize(other.size_);
  }

  // we use mutable_*_data() for 'this' to ensure allocation
  // we use other.*_ptr_ for 'other' so we don't change the value of other.sync_status_

  if (other.host_ptr_) {
    #if HAS_CUDA
    #if SYNCEDMEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyHostToHost" << std::endl;
    #endif
    CHECK_CUDA_STATUS(cudaMemcpy(mutable_host_data(), other.host_ptr_, size_ * sizeof(T), cudaMemcpyHostToHost));
    #else
    memcpy(mutable_host_data(), other.const_host_data(), size_ * sizeof(T));
    #endif
  }

  if (other.device_ptr_) {
    #if SYNCEDMEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
    #endif
    CHECK_CUDA_STATUS(cudaMemcpy(mutable_device_data(), other.device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  sync_status_ = other.sync_status_;
}


} // namespace jams

#endif //JAMS_SYNCED_MEMORY_H
