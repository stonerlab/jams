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
#include <array>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename T>
struct supports_fast_zero : std::bool_constant<std::is_integral_v<T> ||
                                               std::is_floating_point_v<T> ||
                                               std::is_enum_v<T>> {};

template <typename T>
struct supports_fast_zero<std::complex<T>> : supports_fast_zero<T> {};

template <typename T, std::size_t N>
struct supports_fast_zero<std::array<T, N>> : supports_fast_zero<T> {};

template <typename T>
inline constexpr bool supports_fast_zero_v = supports_fast_zero<T>::value;

template <typename T>
bool is_all_zero_representation(const T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>,
                "SyncedMemory zero fast-path requires trivially copyable values");

  unsigned char bytes[sizeof(T)] = {};
  std::memcpy(bytes, &value, sizeof(T));
  return std::all_of(std::begin(bytes), std::end(bytes),
                     [](unsigned char byte) { return byte == 0; });
}
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

    static_assert(std::is_trivially_copyable<T>::value,
      "SyncedMemory<T> requires trivially copyable T (uses memcpy)");

    /// Tracks which side most recently mutated the memory contents.
    enum class LastWriter {
        NONE,   ///< No side currently owns a fresher copy
        HOST,   ///< Host memory was the last side marked mutable
        DEVICE  ///< Device memory was the last side marked mutable
    };

private:
    // DATA
    size_type         size_             = 0;       ///< Number of elements which can be held
    mutable pointer   host_ptr_         = nullptr; ///< Pointer to start of host memory
    mutable pointer   device_ptr_       = nullptr; ///< Pointer to start of GPU memory
    mutable bool      host_valid_       = false;   ///< Whether host memory holds up-to-date contents
    mutable bool      device_valid_     = false;   ///< Whether device memory holds up-to-date contents
    mutable LastWriter last_writer_     = LastWriter::NONE; ///< Which side most recently mutated the data
    mutable bool      host_cuda_malloc_ = false; ///< Whether host memory was allocated was allocated with CudaMalloc

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
    [[nodiscard]] constexpr std::size_t bytes() const noexcept;

    /// Compiler time helper for calculating bytes required for n elements of T
    static constexpr std::size_t bytes(size_type n) noexcept;

    /// Return the maximum allocatable size in number of elements of type 'T'.
    [[nodiscard]] constexpr size_type max_size() const noexcept;

    /// MANIPULATORS
    /// ------------
    void clear() noexcept;

    /// zero all elements of the data
    void zero();

    /// fill all elements of the data with the given value
    void fill(const value_type& value);

    /// resize the data (destructive, reallocates)
    void resize(size_type new_size) noexcept;

    // ACCESSORS

    /// Return const_pointer to start of host data
    const_pointer const_host_data() const;

    /// Return const_point to start of device (GPU) data
    const_pointer const_device_data() const;

    /// Return mutable pointer to start of host data
    pointer mutable_host_data();

    /// Return mutable pointer to start of device (GPU) data
    pointer mutable_device_data();

private:
    /// Return 'true' if this thread has an active CUDA context
    [[nodiscard]] bool has_cuda_context() const noexcept;

    /// Copy host data to the device
    void copy_to_device() const;

    /// Copy device data to the host
    void copy_to_host() const;

    /// Allocate host data with size number of elements
    void allocate_host_memory(size_type size) const;

    /// Allocate device data with size number of elements
    void allocate_device_memory(size_type size) const;

    /// Set device data to zero
    void zero_device() const;

    /// Set host data to zero
    void zero_host() const;

    /// Theoretical maximum number of elements which can be allocated on the host
    [[nodiscard]] constexpr size_type max_size_host() const noexcept;

    /// Theoretical maximum number of elements which can be allocated on the device
    [[nodiscard]] size_type max_size_device() const;

    /// Free memory allocated on the host
    void free_host_memory() noexcept;

    /// Free memory allocated on the device
    void free_device_memory() noexcept;

    template<class InputIt>
    void init_from_range(InputIt first, InputIt last, std::input_iterator_tag);

    template<class ForwardIt>
    void init_from_range(ForwardIt first, ForwardIt last, std::forward_iterator_tag);

    void copy_authoritative_from(const SyncedMemory& rhs);
    void move_from(SyncedMemory&& rhs) noexcept;
    void mark_host_modified() const noexcept;
    void mark_device_modified() const noexcept;
    void mark_synchronized() const noexcept;
    void reset_sync_state() noexcept;
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
  fill(x);
}


template<class T>
template<class InputIt, std::enable_if_t<is_iterator<InputIt>::value, bool>>
SyncedMemory<T>::SyncedMemory(InputIt first, InputIt last)
{
  using iterator_category = typename std::iterator_traits<InputIt>::iterator_category;
  init_from_range(first, last, iterator_category{});
}


template<class T>
SyncedMemory<T>::SyncedMemory(const SyncedMemory &rhs)
    : size_(rhs.size_) {
  copy_authoritative_from(rhs);
}


template<class T>
SyncedMemory<T>::SyncedMemory(SyncedMemory &&rhs) noexcept
    : size_(0) {
  move_from(std::move(rhs));
}


template<class T>
SyncedMemory<T>::~SyncedMemory() {
  free_host_memory();
  free_device_memory();
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(const SyncedMemory& rhs) &{
  if (this != &rhs) {
    SyncedMemory tmp(rhs);
    swap(*this, tmp);
  }
  return *this;
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(SyncedMemory &&rhs) & noexcept {
  if (this != &rhs) {
    free_host_memory();
    free_device_memory();
    size_ = 0;
    reset_sync_state();
    move_from(std::move(rhs));
  }
  return *this;
}


template<class T>
void
SyncedMemory<T>::allocate_device_memory(const SyncedMemory::size_type size) const {
#if HAS_CUDA
  if (size == 0) return;

  if (size > max_size_device()) {
    throw std::bad_alloc();
  }

  // Compile-time check for alignment guarantee of cudaMalloc.
  static_assert(alignof(T) <= 256,
                "SyncedMemory<T>: alignof(T) > 256 may not be satisfied by cudaMalloc alignment");

  assert(!device_ptr_);
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), bytes(size)));
  assert(device_ptr_);
#endif
}


template<class T>
void SyncedMemory<T>::allocate_host_memory(const SyncedMemory::size_type size) const {
  if (size == 0) return;

  // host_ptr_ must not already be allocated before we try to allocate
  assert(!host_ptr_);
  host_cuda_malloc_ = false;

#if HAS_CUDA
  if (has_cuda_context()) {
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMallocHost(reinterpret_cast<void **>(&host_ptr_), bytes(size)));
    assert(host_ptr_);
    host_cuda_malloc_ = true;
    return;
  }
#endif

  // Ensure the returned pointer satisfies alignment requirements for T.
  // posix_memalign requires alignment to be a power of two and a multiple of sizeof(void*).
  const std::size_t alignment = std::max<std::size_t>(SYNCED_MEMORY_HOST_ALIGNMENT, alignof(T));
  static_assert((SYNCED_MEMORY_HOST_ALIGNMENT & (SYNCED_MEMORY_HOST_ALIGNMENT - 1)) == 0,
                "SYNCED_MEMORY_HOST_ALIGNMENT must be a power of two");

  void* raw = nullptr;
  if (posix_memalign(&raw, alignment, bytes(size)) != 0) {
    throw std::bad_alloc();
  }
  host_ptr_ = reinterpret_cast<pointer>(raw);

  // host_ptr_ must be allocated by the end of the function
  assert(host_ptr_);
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_host_data() const {
  copy_to_host();
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_device_data() const {
  copy_to_device();
  return device_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_host_data() {
  copy_to_host();
  mark_host_modified();
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_device_data() {
  copy_to_device();
  mark_device_modified();
  return device_ptr_;
}


template<class T>
void SyncedMemory<T>::copy_to_device() const {
  #if HAS_CUDA
  if (device_valid_) {
    return;
  }

  if (host_valid_) {
    if (!device_ptr_) {
      allocate_device_memory(size_);
    }
    #if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyHostToDevice" << std::endl;
    #endif
    if (size_ != 0) {
      assert(device_ptr_ && host_ptr_);
      SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(device_ptr_, host_ptr_, bytes(size_),
                                   cudaMemcpyHostToDevice));
    }
    mark_synchronized();
    return;
  }

  allocate_device_memory(size_);
  #if SYNCED_MEMORY_ZERO_ON_ALLOCATION
  zero_device();
  #endif
  mark_device_modified();
  #endif
}


template<class T>
void SyncedMemory<T>::copy_to_host() const {
  // Splitting into if statements and returning early has improved performance
  // over using a switch statement (on GCC at least).
  if (host_valid_) {
    return;
  }

  if (device_valid_) {
    #if HAS_CUDA
    if (!host_ptr_) {
      allocate_host_memory(size_);
    }
    #if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToHost" << std::endl;
    #endif
    if (size_ != 0) {
      assert(device_ptr_ && host_ptr_);
      SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(host_ptr_, device_ptr_, bytes(size_), cudaMemcpyDeviceToHost));
    }
    mark_synchronized();
    #endif
    return;
  }

  allocate_host_memory(size_);
  #if SYNCED_MEMORY_ZERO_ON_ALLOCATION
  zero_host();
  #endif
  mark_host_modified();
}


template<class T>
inline
void SyncedMemory<T>::zero_device() const {
  if (size_ == 0) return;

  #if HAS_CUDA
  #if SYNCED_MEMORY_PRINT_MEMSET
  std::cout << "INFO(SyncedMemory): device zero" << std::endl;
  #endif
assert(device_ptr_);
SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemset(device_ptr_, 0, bytes(size_)));
  #endif
}


template<class T>
inline
void SyncedMemory<T>::zero_host() const {
  if (size_ == 0) return;

  #if SYNCED_MEMORY_PRINT_MEMSET
    std::cout << "INFO(SyncedMemory): host zero" << std::endl;
  #endif
  assert(host_ptr_);
  memset(host_ptr_, 0, bytes(size_));
}


template<class T>
void SyncedMemory<T>::zero() {
  static_assert(std::is_default_constructible_v<value_type>,
                "SyncedMemory::zero requires default-constructible value_type");
  fill(value_type{});
}


template<class T>
void SyncedMemory<T>::fill(const value_type& value) {
  if (size_ == 0) {
    return;
  }

  if constexpr (supports_fast_zero_v<value_type>) {
    if (is_all_zero_representation(value)) {
      if (!host_ptr_ && !device_ptr_) {
        allocate_host_memory(size_);
      }
      if (host_ptr_) {
        zero_host();
        mark_host_modified();
      }
      #if HAS_CUDA
      if (device_ptr_) {
        zero_device();
        mark_device_modified();
      }
      if (host_ptr_ && device_ptr_) {
        mark_synchronized();
      }
      #endif
      return;
    }
  }

  pointer p = mutable_host_data();
  std::fill(p, p + size_, value);
}


template<class T>
void SyncedMemory<T>::mark_host_modified() const noexcept {
  host_valid_ = true;
  device_valid_ = false;
  last_writer_ = LastWriter::HOST;
}


template<class T>
void SyncedMemory<T>::mark_device_modified() const noexcept {
  host_valid_ = false;
  device_valid_ = true;
  last_writer_ = LastWriter::DEVICE;
}


template<class T>
void SyncedMemory<T>::mark_synchronized() const noexcept {
  host_valid_ = true;
  device_valid_ = true;
  last_writer_ = LastWriter::NONE;
}


template<class T>
void SyncedMemory<T>::reset_sync_state() noexcept {
  host_valid_ = false;
  device_valid_ = false;
  last_writer_ = LastWriter::NONE;
}


template<class T>
template<class InputIt>
void SyncedMemory<T>::init_from_range(InputIt first, InputIt last, std::input_iterator_tag) {
  const std::vector<value_type> values(first, last);
  size_ = values.size();
  pointer p = mutable_host_data();
  std::copy(values.begin(), values.end(), p);
}


template<class T>
template<class ForwardIt>
void SyncedMemory<T>::init_from_range(ForwardIt first, ForwardIt last, std::forward_iterator_tag) {
  size_ = static_cast<size_type>(std::distance(first, last));
  pointer p = mutable_host_data();
  std::copy(first, last, p);
}


template<class T>
void SyncedMemory<T>::copy_authoritative_from(const SyncedMemory& rhs) {
  if (rhs.size_ == 0) {
    reset_sync_state();
    return;
  }

  const bool prefer_device =
      rhs.device_valid_ && (!rhs.host_valid_ || rhs.last_writer_ == LastWriter::DEVICE);

  if (prefer_device && rhs.device_ptr_) {
#if HAS_CUDA
#if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
#endif
    allocate_device_memory(size_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(
        cudaMemcpy(device_ptr_, rhs.device_ptr_, bytes(size_), cudaMemcpyDeviceToDevice));
    mark_device_modified();
    return;
#endif
  }

  if (rhs.host_valid_ && rhs.host_ptr_) {
    allocate_host_memory(size_);
    std::memcpy(host_ptr_, rhs.host_ptr_, bytes(size_));
    mark_host_modified();
    return;
  }

  if (rhs.device_valid_ && rhs.device_ptr_) {
#if HAS_CUDA
#if SYNCED_MEMORY_PRINT_MEMCPY
    std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
#endif
    allocate_device_memory(size_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(
        cudaMemcpy(device_ptr_, rhs.device_ptr_, bytes(size_), cudaMemcpyDeviceToDevice));
    mark_device_modified();
    return;
#endif
  }

  reset_sync_state();
}


template<class T>
void SyncedMemory<T>::move_from(SyncedMemory&& rhs) noexcept {
  size_ = std::exchange(rhs.size_, 0);
  host_ptr_ = std::exchange(rhs.host_ptr_, nullptr);
  device_ptr_ = std::exchange(rhs.device_ptr_, nullptr);
  host_valid_ = std::exchange(rhs.host_valid_, false);
  device_valid_ = std::exchange(rhs.device_valid_, false);
  last_writer_ = std::exchange(rhs.last_writer_, LastWriter::NONE);
  host_cuda_malloc_ = std::exchange(rhs.host_cuda_malloc_, false);
}


template<class T>
void SyncedMemory<T>::free_host_memory() noexcept {
  if (host_ptr_) {
    #if HAS_CUDA
    if (host_cuda_malloc_) {
      auto status = cudaFreeHost(host_ptr_);
      host_ptr_ = nullptr;
      host_cuda_malloc_ = false;
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
    host_cuda_malloc_ = false;
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
  return max_size_host();
}


template<class T>
void SyncedMemory<T>::resize(SyncedMemory::size_type new_size) noexcept {
  if (size_ == new_size) return;

  size_ = new_size;
  free_host_memory();
  free_device_memory();
  reset_sync_state();
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
  return std::numeric_limits<size_type>::max() / sizeof(value_type);
}


template<class T>
void swap(SyncedMemory<T> &lhs, SyncedMemory<T> &rhs) noexcept {
  using std::swap;
  swap(lhs.size_, rhs.size_);
  swap(lhs.host_ptr_, rhs.host_ptr_);
  swap(lhs.device_ptr_, rhs.device_ptr_);
  swap(lhs.host_valid_, rhs.host_valid_);
  swap(lhs.device_valid_, rhs.device_valid_);
  swap(lhs.last_writer_, rhs.last_writer_);
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
constexpr std::size_t SyncedMemory<T>::bytes() const noexcept {
  return bytes(size_);
}


template<class T>
constexpr std::size_t SyncedMemory<T>::bytes(size_type n) noexcept {
  return n * sizeof(value_type);
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
