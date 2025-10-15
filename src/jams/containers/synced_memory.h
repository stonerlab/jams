// synced_memory.h                                                    -*-C++-*-
#ifndef INCLUDED_JAMS_SYNCED_MEMORY
#define INCLUDED_JAMS_SYNCED_MEMORY
///
/// @purpose: Provide an allocator for synchronised host and CUDA GPU memory
///
/// @classes:
///   jams::SyncedMemory<T>: allocator for host/cuda gpu synchronised memory
///
/// @description: This component provides a concrete allocator,
/// 'jams::SyncedMemory<T>' which lazily allocates memory on the host and CUDA
/// GPU. Accessing host or device pointers will allocate and synchronise the
/// memory spaces performing any needed memory transfers.
///
/// The lazy allocation means that the memory is not allocated until it is
/// accessed for the first time. Therefore if the memory is only ever accessed
/// by the host, no GPU memory is allocated. Moreover, the host memory
/// allocation checks if there is a CUDA device available. If a CUDA device
/// exists then the host memory is allocated with CudaMallocHost as pinned
/// memory for improved performance for data transfers between the GPU and host.
/// If there is no CUDA device we allocated aligned memory.
///
/// NOTE: We previously checked if there is an active CUDA context (i.e. there
/// could be a device present but if there are no CUDA calls made then the
/// context is never initialised). However this causes requires the binary to
/// be linked to libcuda.so (rather than libcudart.so) which is part of the
/// driver, not part of the runtime API. Binaries compiled against libcuda.so
/// therefore don't work on machines without the CUDA driver installed even
/// if they do have the CUDA runtime installed.
///
/// Usage
/// -----

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

// toggle printing all host/device synchronization calls to cout
#define SYNCED_MEMORY_PRINT_MEMCPY 0

// toggle printing all host/device memset calls to cout
#define SYNCED_MEMORY_PRINT_MEMSET 0

// only include <iostream> if we actually use it
#if SYNCED_MEMORY_PRINT_MEMCPY || SYNCED_MEMORY_PRINT_MEMSET
#include <iostream>
#endif

// toggles support for using SyncedMemory in the global namespace
#define SYNCED_MEMORY_ALLOW_GLOBAL 1
//
// If SyncedMemory is used in a global context then free()
// calls CUDA routines after the CUDA context has been unloaded.
// The calls then return cudaErrorCudartUnloading as the status.
// This flag avoids checking the return status in free().

// toggle zeroing of host/device memory immediately after allocation
#define SYNCED_MEMORY_ZERO_ON_ALLOCATION 0

// memory alignment for host memory (if supported)
#define SYNCED_MEMORY_HOST_ALIGNMENT 64

#define SYNCED_MEMORY_CHECK_CUDA_STATUS(x) \
{ \
  cudaError_t stat; \
  if ((stat = (x)) != cudaSuccess) { \
    throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) + " CUDA error: " + cudaGetErrorString(stat)); \
  } \
}

namespace {
template <typename... >
using void_t = void;

template <class T, class = void>
struct is_iterator : std::false_type { };

template <class T>
struct is_iterator<T, void_t<
                      typename std::iterator_traits<T>::iterator_category
>> : std::true_type { };
}

namespace jams {

// ==================
// class SyncedMemory
// ==================
template<class T>
class SyncedMemory {
public:
    // TYPES
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = std::size_t;

    /// Memory contents synchronisation state.
    enum class SyncStatus {
        UNINITIALIZED,     ///< Memory has not been allocated yet
        SYNCHRONIZED,      ///< Host and GPU memory contents are in sync
        DEVICE_IS_MUTATED, ///< GPU memory contents has changed since last sync
        HOST_IS_MUTATED    ///< Host memory contents has changed since last sync
    };

private:
    // DATA
    size_type  size_             = 0;       ///< Number of elements which can be held
    pointer    host_ptr_         = nullptr; ///< Pointer to start of host memory
    pointer    device_ptr_       = nullptr; ///< Pointer to start of GPU memory
    SyncStatus sync_status_      = SyncStatus::UNINITIALIZED; ///< Current synchronisation status
    bool       host_cuda_malloc_ = false; ///< Whether host memory was allocated was allocated with CudaMalloc

public:
    // FRIENDS

    /// Swap two SyncedMemory objects
    template<class F>
    friend void swap(SyncedMemory<F> &lhs, SyncedMemory<F> &rhs) noexcept;

    // CREATORS

    /// Construct a default, uninitialised synced memory object of zero size.
    SyncedMemory() noexcept = default;

    /// Construct a synced memory object with given 'size' (number of elements
    /// of type 'T'.
    explicit SyncedMemory(size_type size) noexcept;

    /// Construct a synced memory object with given 'size' (number of elements
    /// of type 'T'. The memory is initialised to the value 'x'.
    SyncedMemory(size_type size, const T &x);

    /// Construct a synced memory object with a size and values taken from
    /// the range between the 'first' and 'last' input iterators.
    template<class InputIt, std::enable_if_t<is_iterator<InputIt>::value, bool> = true>
    SyncedMemory(InputIt first, InputIt last);

    /// Construct a synced memory object from another similar object.
    SyncedMemory(const SyncedMemory &rhs);

    /// Move constructor
    SyncedMemory(SyncedMemory &&rhs) noexcept;

    /// Destroy the synchronised memory. All memory allocated of the host and
    /// GPU is released.
    ~SyncedMemory();

    /// copy assign
    SyncedMemory &operator=(const SyncedMemory& rhs) &;

    /// move assign
    SyncedMemory &operator=(SyncedMemory &&rhs) & noexcept;

    /// Return the size of the memory allocated as the number of elements of type
    /// 'T'.
    [[nodiscard]] constexpr size_type size() const noexcept;

    /// Return the size of the memory allocated of a single buffer (host or GPU)
    /// in bytes.
    [[nodiscard]] constexpr std::size_t memory() const noexcept;

    /// Return the maximum allocatable size in number of elements of type 'T'.
    [[nodiscard]] constexpr size_type max_size() const noexcept;

    /// MANIPULATORS
    /// ------------
    void clear() noexcept;

    /// zero all elements of the data
    void zero();

    /// resize the data (destructive, reallocates)
    void resize(size_type new_size) noexcept;

    // ACCESSORS

    /// Return const_pointer to start of host data
    const_pointer const_host_data();

    /// Return const_point to start of device (GPU) data
    const_pointer const_device_data();

    /// Return mutable pointer to start of host data
    pointer mutable_host_data();

    /// Return mutable pointer to start of device (GPU) data
    pointer mutable_device_data();

private:
    /// Return 'true' if this thread has an active CUDA context
    [[nodiscard]] bool has_cuda_context() const noexcept;

    /// Copy host data to the device
    void copy_to_device();

    /// Copy device data to the host
    void copy_to_host();

    /// Allocate host data with size number of elements
    void allocate_host_memory(size_type size);

    /// Allocate device data with size number of elements
    void allocate_device_memory(size_type size);

    /// Set device data to zero
    void zero_device();

    /// Set host data to zero
    void zero_host();

    /// Theoretical maximum number of elements which can be allocated on the host
    [[nodiscard]] constexpr size_type max_size_host() const noexcept;

    /// Theoretical maximum number of elements which can be allocated on the device
    [[nodiscard]] size_type max_size_device() const;

    /// Free memory allocated on the host
    void free_host_memory() noexcept;

    /// Free memory allocated on the device
    void free_device_memory() noexcept;
};

// ============================================================================
//                      INLINE FUNCTION DEFINITIONS
// ============================================================================

template<class T>
SyncedMemory<T>::SyncedMemory(SyncedMemory::size_type size) noexcept
    : size_(size) {}


template<class T>
SyncedMemory<T>::SyncedMemory(SyncedMemory::size_type size, const T &x)
    : size_(size) {
  if (x == T{0}) {
    zero();
  } else {
    std::fill(mutable_host_data(), mutable_host_data() + size_, x);
  }
}


template<class T>
template<class InputIt, std::enable_if_t<is_iterator<InputIt>::value, bool>>
SyncedMemory<T>::SyncedMemory(InputIt first, InputIt last)
    : size_(std::distance(first, last)) {
  std::copy(first, last, mutable_host_data());
}


template<class T>
SyncedMemory<T>::SyncedMemory(const SyncedMemory &rhs)
    : sync_status_(rhs.sync_status_) {
  if (size_ != rhs.size_) {
    resize(rhs.size_);
  }

  // We use mutable_*_data() for 'this' to ensure allocation
  // We use rhs.*_ptr_ for 'rhs' so we don't change the value of rhs.sync_status_
  if (rhs.host_ptr_) {
    if (has_cuda_context()) {
#if HAS_CUDA
      // 2021-03-16 Joe: The use of cudaMemcpy(..., cudaMemcpyHostToHost)
      // may not be strictly necessary. Just using memcpy works because
      // the pointers are all host pointers. However cudaMemcpy should be
      // enforcing device synchronisation so that we don't copy host memory
      // while it's being asynchronously copied into or out of else where.
#if SYNCED_MEMORY_PRINT_MEMCPY
      std::cout << "INFO(SyncedMemory): cudaMemcpyHostToHost" << std::endl;
#endif
      SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(mutable_host_data(), rhs.host_ptr_, size_ * sizeof(T), cudaMemcpyHostToHost));
#endif
    } else {
      memcpy(mutable_host_data(), rhs.host_ptr_, size_ * sizeof(T));
    }
  }

  if (rhs.device_ptr_) {
#if HAS_CUDA
#if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
#endif
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(mutable_device_data(), rhs.device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
#endif
  }
}


template<class T>
SyncedMemory<T>::SyncedMemory(SyncedMemory &&rhs) noexcept
    : size_(std::move(rhs.size_))
    , host_ptr_(std::move(rhs.host_ptr_))
    , device_ptr_(std::move(rhs.device_ptr_))
    , sync_status_(std::move(rhs.sync_status_))
    , host_cuda_malloc_(std::move(rhs.host_cuda_malloc_)){
  rhs.sync_status_ = SyncStatus::UNINITIALIZED;
  rhs.size_ = 0;
  rhs.host_ptr_ = nullptr;
  rhs.device_ptr_ = nullptr;
}


template<class T>
SyncedMemory<T>::~SyncedMemory() {
  free_host_memory();
  free_device_memory();
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(const SyncedMemory& rhs) &{
  if (this != &rhs) {
    // Only reallocate if sizes are different and only resize the memory
    // spaces that were already allocated on the rhs.
    if (size_ != rhs.size_) {
      if (rhs.host_ptr_) {
        free_host_memory();
        allocate_host_memory(rhs.size_);
      }

      if (rhs.device_ptr_) {
        free_device_memory();
        allocate_device_memory(rhs.size_);
      }
    }

    size_ = rhs.size_;

    // we use mutable_*_data() for 'this' to ensure allocation
    // we use rhs.*_ptr_ for 'rhs' so we don't change the value of rhs.sync_status_

    if (rhs.host_ptr_) {
      #if HAS_CUDA
      if (has_cuda_context()) {
        #if SYNCED_MEMORY_PRINT_MEMCPY
        std::cout << "INFO(SyncedMemory): cudaMemcpyHostToHost" << std::endl;
        #endif
        SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(mutable_host_data(), rhs.host_ptr_, size_ * sizeof(T), cudaMemcpyHostToHost));
      } else {
        memcpy(mutable_host_data(), rhs.host_ptr_, size_ * sizeof(T));
      }
      #else
      memcpy(mutable_host_data(), rhs.host_ptr_, size_ * sizeof(T));
      #endif
    }

    if (rhs.device_ptr_) {
      #if HAS_CUDA
      #if SYNCED_MEMORY_PRINT_MEMCPY
      std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
      #endif
      SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(mutable_device_data(), rhs.device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
      #endif
    }
  }
  return *this;
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(SyncedMemory &&rhs) & noexcept {
  size_ = rhs.size_;
  host_ptr_ = rhs.host_ptr_;
  device_ptr_ = rhs.device_ptr_;
  sync_status_ = rhs.sync_status_;
  host_cuda_malloc_ = rhs.host_cuda_malloc_;
  rhs.sync_status_ = SyncStatus::UNINITIALIZED;
  rhs.size_ = 0;
  rhs.host_ptr_ = nullptr;
  rhs.device_ptr_ = nullptr;
  rhs.host_cuda_malloc_ = false;
  return *this;
}


template<class T>
void
SyncedMemory<T>::allocate_device_memory(const SyncedMemory::size_type size) {
  #if HAS_CUDA
  if (size == 0) return;

  if (size > max_size_device()) {
    throw std::bad_alloc();
  }

  assert(!device_ptr_);
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), size * sizeof(T)));
  assert(device_ptr_);
  size_ = size;
  #endif
}


template<class T>
void SyncedMemory<T>::allocate_host_memory(const SyncedMemory::size_type size) {
  if (size == 0) return;

  // host_ptr_ must not already be allocated before we try to allocate
  assert(!host_ptr_);

  #if HAS_CUDA
  if (has_cuda_context()) {
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMallocHost(reinterpret_cast<void **>(&host_ptr_), size * sizeof(T)));
    assert(host_ptr_);
    size_ = size;
    host_cuda_malloc_ = true;
    return;
  }
  #endif

  if (posix_memalign(reinterpret_cast<void **>(&host_ptr_),
                     SYNCED_MEMORY_HOST_ALIGNMENT,
                     size * sizeof(T)) != 0) {
    throw std::bad_alloc();
  }

  // host_ptr_ must be allocated by the end of the function
  assert(host_ptr_);
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_host_data() {
  copy_to_host();
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_device_data() {
  copy_to_device();
  return device_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_host_data() {
  copy_to_host();
  sync_status_ = SyncStatus::HOST_IS_MUTATED;
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_device_data() {
  copy_to_device();
  sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
  return device_ptr_;
}


template<class T>
void SyncedMemory<T>::copy_to_device() {
  #if HAS_CUDA
  if (sync_status_ == SyncStatus::DEVICE_IS_MUTATED || sync_status_ == SyncStatus::SYNCHRONIZED) {
    return;
  }

  if (sync_status_ == SyncStatus::HOST_IS_MUTATED) {
    if (!device_ptr_) {
      allocate_device_memory(size_);
    }
    #if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyHostToDevice" << std::endl;
    #endif
    assert(device_ptr_ && host_ptr_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(device_ptr_, host_ptr_, size_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
    sync_status_ = SyncStatus::SYNCHRONIZED;
    return;
  }

  if (sync_status_ == SyncStatus::UNINITIALIZED) {
    allocate_device_memory(size_);
    #if SYNCED_MEMORY_ZERO_ON_ALLOCATION
    zero_device();
    #endif
    sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
    return;
  }
  #endif
}


template<class T>
void SyncedMemory<T>::copy_to_host() {
  // Splitting into if statements and returning early has improved performance
  // over using a switch statement (on GCC at least).
  if (sync_status_ == SyncStatus::HOST_IS_MUTATED || sync_status_ == SyncStatus::SYNCHRONIZED) {
    return;
  }

  if (sync_status_ == SyncStatus::DEVICE_IS_MUTATED) {
    #if HAS_CUDA
    if (!host_ptr_) {
      allocate_host_memory(size_);
    }
    #if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToHost" << std::endl;
    #endif
    assert(device_ptr_ && host_ptr_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(host_ptr_, device_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    sync_status_ = SyncStatus::SYNCHRONIZED;
    #endif
    return;
  }

  if (sync_status_ == SyncStatus::UNINITIALIZED) {
    allocate_host_memory(size_);
    #if SYNCED_MEMORY_ZERO_ON_ALLOCATION
    zero_host();
    #endif
    sync_status_ = SyncStatus::HOST_IS_MUTATED;
    return;
  }
}


template<class T>
inline
void SyncedMemory<T>::zero_device() {
  if (size_ == 0) return;

  #if HAS_CUDA
  #if SYNCED_MEMORY_PRINT_MEMSET
  std::cout << "INFO(SyncedMemory): device zero" << std::endl;
  #endif
assert(device_ptr_);
SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemset(device_ptr_, 0, size_ * sizeof(T)));
  #endif
}


template<class T>
inline
void SyncedMemory<T>::zero_host() {
  if (size_ == 0) return;

  #if SYNCED_MEMORY_PRINT_MEMSET
    std::cout << "INFO(SyncedMemory): host zero" << std::endl;
  #endif
  assert(host_ptr_);
  memset(host_ptr_, 0, size_ * sizeof(T));
}


template<class T>
void SyncedMemory<T>::zero() {
  if (!host_ptr_ && !device_ptr_) {
    allocate_host_memory(size_);
  }
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
void SyncedMemory<T>::free_host_memory() noexcept {
  if (host_ptr_) {
    #if HAS_CUDA
    if (host_cuda_malloc_) {
      auto status = cudaFreeHost(host_ptr_);
      host_ptr_ = nullptr;
      #if SYNCED_MEMORY_ALLOW_GLOBAL
      assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
      #else
      assert(status == cudaSuccess);
      #endif
      return;
    }
    #endif
    free(host_ptr_);
    host_ptr_ = nullptr;
  }
}


template<class T>
void SyncedMemory<T>::free_device_memory() noexcept{
  #if HAS_CUDA
  if (device_ptr_) {
    auto status = cudaFree(device_ptr_);
    #if SYNCED_MEMORY_ALLOW_GLOBAL
    assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
    #else
    assert(status == cudaSuccess);
    #endif
  }
  #endif
  device_ptr_ = nullptr;
}


template<class T>
constexpr
typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size() const noexcept {
  #if HAS_CUDA
  return std::min(max_size_host(), max_size_device());
  #else
  return max_size_host();
  #endif
}


template<class T>
void SyncedMemory<T>::resize(SyncedMemory::size_type new_size) noexcept {
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
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, dev));
  return prop.maxGridSize[0];
  #else
  return 0;
  #endif
}


template<class T>
constexpr typename SyncedMemory<T>::size_type
SyncedMemory<T>::max_size_host() const noexcept {
  return std::numeric_limits<size_type>::max();
}


template<class T>
void swap(SyncedMemory<T> &lhs, SyncedMemory<T> &rhs) noexcept {
  using std::swap;
  swap(lhs.sync_status_, rhs.sync_status_);
  swap(lhs.size_, rhs.size_);
  swap(lhs.host_ptr_, rhs.host_ptr_);
  swap(lhs.device_ptr_, rhs.device_ptr_);
  swap(lhs.host_cuda_malloc_, rhs.host_cuda_malloc_);
}


template<class T>
bool SyncedMemory<T>::has_cuda_context() const noexcept{
#if HAS_CUDA
  int device;
  cudaError_t status = cudaGetDevice(&device);
  return status == cudaSuccess;
#else
  return false;
#endif
}


template<class T>
constexpr typename SyncedMemory<T>::size_type
SyncedMemory<T>::size() const noexcept { return size_; }


template<class T>
constexpr std::size_t SyncedMemory<T>::memory() const noexcept {
  return size_ * sizeof(value_type);
}


template<class T>
void SyncedMemory<T>::clear() noexcept { resize(0); }


} // namespace jams


#undef SYNCED_MEMORY_PRINT_MEMCPY
#undef SYNCED_MEMORY_PRINT_MEMSET
#undef SYNCED_MEMORY_ALLOW_GLOBAL
#undef SYNCED_MEMORY_ZERO_ON_ALLOCATION
#undef SYNCED_MEMORY_HOST_ALIGNMENT
#undef SYNCED_MEMORY_CHECK_CUDA_STATUS

#endif
// ----------------------------- END-OF-FILE ----------------------------------