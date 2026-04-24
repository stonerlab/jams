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
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace jams::detail {
inline constexpr bool synced_memory_print_memcpy = false;
inline constexpr bool synced_memory_print_memset = false;
inline constexpr bool synced_memory_zero_on_allocation = false;
inline constexpr std::size_t synced_memory_host_alignment = 64;

// If SyncedMemory is used in a global context then free() can call CUDA
// routines after the CUDA context has been unloaded.
inline constexpr bool synced_memory_allow_global = true;

#if HAS_CUDA
inline void check_cuda_status(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             " CUDA error: " + cudaGetErrorString(status));
  }
}

[[noreturn]] inline void terminate_cuda_free_failure(cudaError_t status, const char* operation) noexcept {
  std::cerr << "FATAL(SyncedMemory): " << operation
            << " failed during noexcept cleanup: "
            << cudaGetErrorString(status) << std::endl;
  std::terminate();
}
#endif

template <class T, class = void>
struct is_iterator : std::false_type { };

template <class T>
struct is_iterator<T, std::void_t<
                      typename std::iterator_traits<T>::iterator_category
>> : std::true_type { };

template <class T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;
} // namespace jams::detail

#if HAS_CUDA
#define SYNCED_MEMORY_CHECK_CUDA_STATUS(x) ::jams::detail::check_cuda_status((x), __FILE__, __LINE__)
#endif

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

    static_assert(std::is_trivially_copyable_v<T>,
      "SyncedMemory<T> requires trivially copyable T (uses memcpy)");

private:
    enum class Validity : unsigned {
        none   = 0,
        host   = 1u << 0u,
        device = 1u << 1u
    };

    // DATA
    size_type        size_             = 0;       ///< Number of elements which can be held
    mutable pointer  host_ptr_         = nullptr; ///< Pointer to start of host memory
    mutable pointer  device_ptr_       = nullptr; ///< Pointer to start of GPU memory
    mutable Validity valid_            = Validity::none; ///< Valid host/device copies.
    mutable bool     host_cuda_malloc_ = false; ///< Whether host memory was allocated with CudaMalloc

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
    template<class InputIt, std::enable_if_t<detail::is_iterator_v<InputIt>, bool> = true>
    SyncedMemory(InputIt first, InputIt last);

    /// Construct a synced memory object from another similar object.
    SyncedMemory(const SyncedMemory &rhs);

    /// Move constructor
    SyncedMemory(SyncedMemory &&rhs) noexcept;

    /// Destroy the synchronised memory. All memory allocated of the host and
    /// GPU is released.
    ~SyncedMemory() noexcept;

    /// copy assign
    SyncedMemory &operator=(const SyncedMemory& rhs) &;

    /// move assign
    SyncedMemory &operator=(SyncedMemory &&rhs) & noexcept;

    /// Return the size of the memory allocated as the number of elements of type
    /// 'T'.
    [[nodiscard]] constexpr size_type size() const noexcept;

    /// Return the size of the memory allocated of a single buffer (host or GPU)
    /// in bytes.
    [[nodiscard]] constexpr std::size_t bytes() const;

    /// Compiler time helper for calculating bytes required for n elements of T
    static constexpr std::size_t bytes(size_type n);

    /// Return the maximum allocatable size in number of elements of type 'T'.
    [[nodiscard]] size_type max_size() const;

    /// MANIPULATORS
    /// ------------
    void clear() noexcept;

    /// zero all elements of the data
    void zero();

    /// resize the data (destructive, reallocates)
    void resize(size_type new_size) noexcept;

    // ACCESSORS

    /// Return true if the host memory contains current data.
    [[nodiscard]] bool host_valid() const noexcept;

    /// Return true if the device memory contains current data.
    [[nodiscard]] bool device_valid() const noexcept;

    /// Return true if size() is zero.
    [[nodiscard]] constexpr bool empty() const noexcept;

    /// Return const_pointer to start of host data.
    const_pointer host_data() const;

    /// Return const_pointer to start of device (GPU) data.
    const_pointer device_data() const;

    /// Compatibility alias for host_data().
    const_pointer const_host_data() const;

    /// Compatibility alias for device_data().
    const_pointer const_device_data() const;

    /// Return mutable pointer to start of host data
    pointer mutable_host_data();

    /// Return mutable pointer to start of device (GPU) data
    pointer mutable_device_data();

private:
    /// Return 'true' if this thread has an active CUDA context
    [[nodiscard]] bool has_cuda_context() const noexcept;

    /// Ensure host memory exists and contains current data.
    void ensure_host_current() const;

    /// Ensure device memory exists and contains current data.
    void ensure_device_current() const;

    /// Mark host memory as the only mutable/current side.
    void mark_host_modified() noexcept;

    /// Mark device memory as the only mutable/current side.
    void mark_device_modified() noexcept;

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
    void free_host_memory() const noexcept;

    /// Free memory allocated on the device
    void free_device_memory() const noexcept;

#if HAS_CUDA
    static bool is_acceptable_free_status(cudaError_t status) noexcept;
#endif

    static constexpr unsigned bits(Validity validity) noexcept;

    [[nodiscard]] bool valid_contains(Validity validity) const noexcept;

    void set_valid(Validity validity) const noexcept;

    void add_valid(Validity validity) const noexcept;

    template<class InputIt>
    void assign_from_input_range(InputIt first, InputIt last);

    template<class ForwardIt>
    void assign_from_forward_range(ForwardIt first, ForwardIt last);
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
    pointer p = mutable_host_data();
    std::fill(p, p + size_, x);
  }
}


template<class T>
template<class InputIt, std::enable_if_t<detail::is_iterator_v<InputIt>, bool>>
SyncedMemory<T>::SyncedMemory(InputIt first, InputIt last) {
  using category = typename std::iterator_traits<InputIt>::iterator_category;
  if constexpr (std::is_base_of_v<std::forward_iterator_tag, category>) {
    assign_from_forward_range(first, last);
  } else {
    assign_from_input_range(first, last);
  }
}


template<class T>
SyncedMemory<T>::SyncedMemory(const SyncedMemory &rhs)
    : size_(rhs.size_) {
  if (rhs.host_valid()) {
    allocate_host_memory(size_);
    std::memcpy(host_ptr_, rhs.host_ptr_, bytes(size_));
    set_valid(Validity::host);
    return;
  }

#if HAS_CUDA
  if (rhs.device_valid()) {
    allocate_device_memory(size_);
    if constexpr (detail::synced_memory_print_memcpy) {
      std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToDevice" << std::endl;
    }
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(device_ptr_, rhs.device_ptr_, bytes(size_), cudaMemcpyDeviceToDevice));
    set_valid(Validity::device);
  }
#endif
}


template<class T>
SyncedMemory<T>::SyncedMemory(SyncedMemory &&rhs) noexcept
    : size_(std::exchange(rhs.size_, 0))
    , host_ptr_(std::exchange(rhs.host_ptr_, nullptr))
    , device_ptr_(std::exchange(rhs.device_ptr_, nullptr))
    , valid_(std::exchange(rhs.valid_, Validity::none))
    , host_cuda_malloc_(std::exchange(rhs.host_cuda_malloc_, false)) {
}


template<class T>
SyncedMemory<T>::~SyncedMemory() noexcept {
  free_host_memory();
  free_device_memory();
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(const SyncedMemory& rhs) &{
  if (this != &rhs) {
    if (size_ != rhs.size_) {
      SyncedMemory tmp(rhs);
      swap(*this, tmp);
      return *this;
    }

    if (rhs.host_valid()) {
      if (!host_ptr_) {
        allocate_host_memory(size_);
      }
      std::memcpy(host_ptr_, rhs.host_ptr_, bytes(size_));
      set_valid(Validity::host);
      return *this;
    }

#if HAS_CUDA
    if (rhs.device_valid()) {
      SyncedMemory tmp(rhs);
      free_device_memory();
      device_ptr_ = std::exchange(tmp.device_ptr_, nullptr);
      set_valid(Validity::device);
      return *this;
    }
#endif

    set_valid(Validity::none);
  }
  return *this;
}


template<class T>
SyncedMemory<T> &SyncedMemory<T>::operator=(SyncedMemory &&rhs) & noexcept {
  if (this == &rhs) {
    return *this;
  }

  free_host_memory();
  free_device_memory();

  size_ = std::exchange(rhs.size_, 0);
  host_ptr_ = std::exchange(rhs.host_ptr_, nullptr);
  device_ptr_ = std::exchange(rhs.device_ptr_, nullptr);
  valid_ = std::exchange(rhs.valid_, Validity::none);
  host_cuda_malloc_ = std::exchange(rhs.host_cuda_malloc_, false);
  return *this;
}


template<class T>
void
SyncedMemory<T>::allocate_device_memory(const SyncedMemory::size_type size) const {
#if HAS_CUDA
  if (size == 0) return;

  const std::size_t allocation_bytes = bytes(size);

  // Compile-time check for alignment guarantee of cudaMalloc.
  static_assert(alignof(T) <= 256,
                "SyncedMemory<T>: alignof(T) > 256 may not be satisfied by cudaMalloc alignment");

  assert(!device_ptr_);
  void* raw = nullptr;
  const cudaError_t status = cudaMalloc(&raw, allocation_bytes);
  if (status == cudaErrorMemoryAllocation) {
    throw std::bad_alloc();
  }
  SYNCED_MEMORY_CHECK_CUDA_STATUS(status);
  device_ptr_ = static_cast<pointer>(raw);
  assert(device_ptr_);
#else
  if (size != 0) {
    throw std::runtime_error("SyncedMemory: CUDA device memory requested but CUDA support is disabled");
  }
#endif
}


template<class T>
void SyncedMemory<T>::allocate_host_memory(const SyncedMemory::size_type size) const {
  if (size == 0) return;

  const std::size_t allocation_bytes = bytes(size);

  // host_ptr_ must not already be allocated before we try to allocate
  assert(!host_ptr_);

#if HAS_CUDA
  if (has_cuda_context()) {
    void* raw = nullptr;
    const cudaError_t status = cudaMallocHost(&raw, allocation_bytes);
    if (status == cudaErrorMemoryAllocation) {
      throw std::bad_alloc();
    }
    SYNCED_MEMORY_CHECK_CUDA_STATUS(status);
    host_ptr_ = static_cast<pointer>(raw);
    assert(host_ptr_);
    host_cuda_malloc_ = true;
    return;
  }
#endif

  // Ensure the returned pointer satisfies alignment requirements for T.
  // posix_memalign requires alignment to be a power of two and a multiple of sizeof(void*).
  constexpr std::size_t host_alignment = detail::synced_memory_host_alignment;
  const std::size_t alignment = std::max<std::size_t>(host_alignment, alignof(T));
  static_assert((host_alignment & (host_alignment - 1)) == 0,
                "synced_memory_host_alignment must be a power of two");

  void* raw = nullptr;
  if (posix_memalign(&raw, alignment, allocation_bytes) != 0) {
    throw std::bad_alloc();
  }
  host_ptr_ = reinterpret_cast<pointer>(raw);
  host_cuda_malloc_ = false;

  // host_ptr_ must be allocated by the end of the function
  assert(host_ptr_);
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::host_data() const {
  ensure_host_current();
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::device_data() const {
  ensure_device_current();
  return device_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_host_data() const {
  return host_data();
}


template<class T>
inline
typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_device_data() const {
  return device_data();
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_host_data() {
  ensure_host_current();
  mark_host_modified();
  return host_ptr_;
}


template<class T>
inline
typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_device_data() {
  ensure_device_current();
  mark_device_modified();
  return device_ptr_;
}


template<class T>
void SyncedMemory<T>::ensure_host_current() const {
  if (host_valid()) {
    return;
  }

  if (size_ == 0) {
    return;
  }

  if (device_valid()) {
#if HAS_CUDA
    if (!host_ptr_) {
      allocate_host_memory(size_);
    }
    if constexpr (detail::synced_memory_print_memcpy) {
      std::cout << "INFO(SyncedMemory): cudaMemcpyDeviceToHost" << std::endl;
    }
    assert(device_ptr_ && host_ptr_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(host_ptr_, device_ptr_, bytes(size_), cudaMemcpyDeviceToHost));
    add_valid(Validity::host);
#endif
    return;
  }

  if (!host_ptr_) {
    allocate_host_memory(size_);
  }
  if constexpr (detail::synced_memory_zero_on_allocation) {
    if (host_ptr_) {
      zero_host();
    }
  }
  if (host_ptr_) {
    add_valid(Validity::host);
  }
}


template<class T>
void SyncedMemory<T>::ensure_device_current() const {
#if HAS_CUDA
  if (device_valid()) {
    return;
  }

  if (size_ == 0) {
    return;
  }

  if (!device_ptr_) {
    allocate_device_memory(size_);
  }

  if (host_valid()) {
    if constexpr (detail::synced_memory_print_memcpy) {
      std::cout << "INFO(SyncedMemory): cudaMemcpyHostToDevice" << std::endl;
    }
    assert(device_ptr_ && host_ptr_);
    SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemcpy(device_ptr_, host_ptr_, bytes(size_),
                                               cudaMemcpyHostToDevice));
    add_valid(Validity::device);
    return;
  }

  if constexpr (detail::synced_memory_zero_on_allocation) {
    zero_device();
  }
  if (device_ptr_) {
    add_valid(Validity::device);
  }
#else
  if (size_ != 0) {
    throw std::runtime_error("SyncedMemory: CUDA device memory requested but CUDA support is disabled");
  }
#endif
}


template<class T>
void SyncedMemory<T>::mark_host_modified() noexcept {
  if (host_ptr_) {
    set_valid(Validity::host);
  } else {
    set_valid(Validity::none);
  }
}


template<class T>
void SyncedMemory<T>::mark_device_modified() noexcept {
  if (device_ptr_) {
    set_valid(Validity::device);
  } else {
    set_valid(Validity::none);
  }
}


template<class T>
inline
void SyncedMemory<T>::zero_device() const {
  if (size_ == 0) return;

#if HAS_CUDA
  if constexpr (detail::synced_memory_print_memset) {
    std::cout << "INFO(SyncedMemory): device zero" << std::endl;
  }
  assert(device_ptr_);
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemset(device_ptr_, 0, bytes(size_)));
#endif
}


template<class T>
inline
void SyncedMemory<T>::zero_host() const {
  if (size_ == 0) return;

  if constexpr (detail::synced_memory_print_memset) {
    std::cout << "INFO(SyncedMemory): host zero" << std::endl;
  }
  assert(host_ptr_);
  memset(host_ptr_, 0, bytes(size_));
}


template<class T>
void SyncedMemory<T>::zero() {
  if (!host_ptr_ && !device_ptr_) {
    allocate_host_memory(size_);
  }
  if (host_ptr_) {
    zero_host();
    add_valid(Validity::host);
  }
  #if HAS_CUDA
  if (device_ptr_) {
    zero_device();
    add_valid(Validity::device);
  }
  #endif
}


template<class T>
void SyncedMemory<T>::free_host_memory() const noexcept {
  if (host_ptr_) {
#if HAS_CUDA
    if (host_cuda_malloc_) {
      if (auto status = cudaFreeHost(host_ptr_); !is_acceptable_free_status(status)) {
        assert(false);
        detail::terminate_cuda_free_failure(status, "cudaFreeHost");
      }
      host_ptr_ = nullptr;
      host_cuda_malloc_ = false;
      return;
    }
    #endif
    free(host_ptr_);
    host_ptr_ = nullptr;
  }
  host_cuda_malloc_ = false;
}


template<class T>
void SyncedMemory<T>::free_device_memory() const noexcept{
  #if HAS_CUDA
  if (device_ptr_) {
    if (auto status = cudaFree(device_ptr_); !is_acceptable_free_status(status)) {
      assert(false);
      detail::terminate_cuda_free_failure(status, "cudaFree");
    }
  }
  #endif
  device_ptr_ = nullptr;
}


template<class T>
typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size() const {
  #if HAS_CUDA
  if (has_cuda_context()) {
    return std::min(max_size_host(), max_size_device());
  }
  #else
  return max_size_host();
  #endif
  return max_size_host();
}


template<class T>
void SyncedMemory<T>::resize(SyncedMemory::size_type new_size) noexcept {
  if (size_ == new_size) return;

  size_ = new_size;
  free_host_memory();
  free_device_memory();
  valid_ = Validity::none;
}


// max_size is for returning the maximum theoretical size a container can be,
// i.e. it is still possible that we cannot allocate this amount of memory.
template<class T>
typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size_device() const {
  #if HAS_CUDA
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  SYNCED_MEMORY_CHECK_CUDA_STATUS(cudaMemGetInfo(&free_bytes, &total_bytes));
  return std::min(max_size_host(), free_bytes / sizeof(value_type));
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
  swap(lhs.valid_, rhs.valid_);
  swap(lhs.size_, rhs.size_);
  swap(lhs.host_ptr_, rhs.host_ptr_);
  swap(lhs.device_ptr_, rhs.device_ptr_);
  swap(lhs.host_cuda_malloc_, rhs.host_cuda_malloc_);
}


template<class T>
constexpr unsigned SyncedMemory<T>::bits(Validity validity) noexcept {
  return static_cast<unsigned>(validity);
}

#if HAS_CUDA
template<class T>
bool SyncedMemory<T>::is_acceptable_free_status(cudaError_t status) noexcept {
  if constexpr (detail::synced_memory_allow_global) {
    return status == cudaSuccess || status == cudaErrorCudartUnloading;
  }
  return status == cudaSuccess;
}
#endif


template<class T>
bool SyncedMemory<T>::valid_contains(Validity validity) const noexcept {
  return (bits(valid_) & bits(validity)) != 0;
}


template<class T>
void SyncedMemory<T>::set_valid(Validity validity) const noexcept {
  valid_ = validity;
}


template<class T>
void SyncedMemory<T>::add_valid(Validity validity) const noexcept {
  valid_ = static_cast<Validity>(bits(valid_) | bits(validity));
}


template<class T>
bool SyncedMemory<T>::host_valid() const noexcept {
  return valid_contains(Validity::host);
}


template<class T>
bool SyncedMemory<T>::device_valid() const noexcept {
  return valid_contains(Validity::device);
}


template<class T>
bool SyncedMemory<T>::has_cuda_context() const noexcept{
#if HAS_CUDA
  if (int device = 0; cudaGetDevice(&device) == cudaSuccess) {
    return true;
  }
  return false;
#else
  return false;
#endif
}


template<class T>
constexpr typename SyncedMemory<T>::size_type
SyncedMemory<T>::size() const noexcept { return size_; }


template<class T>
constexpr bool SyncedMemory<T>::empty() const noexcept {
  return size_ == 0;
}


template<class T>
constexpr std::size_t SyncedMemory<T>::bytes() const {
  return bytes(size_);
}


template<class T>
constexpr std::size_t SyncedMemory<T>::bytes(size_type n) {
  if (n > std::numeric_limits<std::size_t>::max() / sizeof(value_type)) {
    throw std::overflow_error("SyncedMemory::bytes size overflow");
  }
  return n * sizeof(value_type);
}

template<class T>
void SyncedMemory<T>::clear() noexcept { resize(0); }


template<class T>
template<class InputIt>
void SyncedMemory<T>::assign_from_input_range(InputIt first, InputIt last) {
  std::vector<T> values(first, last);
  size_ = values.size();
  if (size_ == 0) {
    return;
  }

  allocate_host_memory(size_);
  std::copy(values.begin(), values.end(), host_ptr_);
  set_valid(Validity::host);
}


template<class T>
template<class ForwardIt>
void SyncedMemory<T>::assign_from_forward_range(ForwardIt first, ForwardIt last) {
  size_ = static_cast<size_type>(std::distance(first, last));
  if (size_ == 0) {
    return;
  }

  allocate_host_memory(size_);
  std::copy(first, last, host_ptr_);
  set_valid(Validity::host);
}


} // namespace jams


#undef SYNCED_MEMORY_CHECK_CUDA_STATUS

#endif
// ----------------------------- END-OF-FILE ----------------------------------
