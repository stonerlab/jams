//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_SYNCED_MEMORY_H
#define JAMS_SYNCED_MEMORY_H

#include <limits>

#if HAS_CUDA
#include <cuda_runtime.h>
#include "jams/cuda/cuda_common.h"
#endif


// PRINT_MEMCPY toggles the printing to cout of all host/device
// sychronization calls within SyncedMemory.
#define PRINT_MEMCPY 0

// PRINT_MEMSET toggles the printing to cout of all host/device
// sychronization calls within SyncedMemory.
#define PRINT_MEMSET 0

// If a SyncedMemory is used in a global context then free()
// calls CUDA routines after the CUDA context has been unloaded.
// The calls then return cudaErrorCudartUnloading as the status.
// ALLOW_GLOBAL_SYNCEDMEMORY ignores this return status in the
// calls to free()
#define ALLOW_GLOBAL_SYNCEDMEMORY 1

// Query the device on memory allocation to check there is
// enough memory available to allocate the requested size
#define DO_FREE_MEMORY_CHECKING 1

namespace jams {
    template <class T>
    class SyncedMemory {
    public:
        typedef T                   value_type;
        typedef T&                  reference;
        typedef const T&            const_reference;
        typedef T*                  pointer;
        typedef const T*            const_pointer;
        typedef size_t              size_type;

        enum class SyncStatus {
            UNINITIALIZED,
            SYNCHRONIZED,
            DEVICE_IS_MUTATED,
            HOST_IS_MUTATED
        };

        SyncedMemory() = default;
        SyncedMemory(const SyncedMemory&) = delete;
        SyncedMemory& operator=(const SyncedMemory&) = delete;

        explicit SyncedMemory(size_type size);
        SyncedMemory(size_type size, const T& x);

        ~SyncedMemory();
        inline constexpr size_type size() const noexcept { return size_; }
        inline constexpr size_type memory() const noexcept { return size_ * sizeof(value_type); }
        inline size_type max_size() const noexcept;

        // Accessors
        inline const_pointer const_host_data();
        inline const_pointer const_device_data();

        inline pointer mutable_host_data();
        inline pointer mutable_device_data();

        // Modifiers
        inline void clear() noexcept { resize(0); }
        inline void zero();
        inline void resize(size_type size);

    private:

        void copy_to_device();
        void copy_to_host();

        void allocate_host_memory(size_type size);
        void allocate_device_memory(size_type size);

        inline void zero_device() noexcept;
        inline void zero_host() noexcept;

        inline constexpr size_type max_size_host() const noexcept;
        inline size_type max_size_device() const noexcept;
        inline size_type available_size_device() const noexcept;

        inline void free_host_memory() noexcept;
        inline void free_device_memory() noexcept;

        size_type size_         = 0;
        pointer host_ptr_       = nullptr;
        pointer device_ptr_     = nullptr;
        SyncStatus sync_status_ = SyncStatus::UNINITIALIZED;
    };

    template<class T>
    inline SyncedMemory<T>::SyncedMemory(SyncedMemory::size_type size)
        : size_(size) {}

    template<class T>
    inline SyncedMemory<T>::SyncedMemory(SyncedMemory::size_type size, const T &x)
        : size_(size) {
      std::fill(mutable_host_data(), mutable_host_data() + size_, x);
    }

    template<class T>
    SyncedMemory<T>::~SyncedMemory() {
      free_host_memory();
      free_device_memory();
    }

    template<class T>
    void SyncedMemory<T>::allocate_device_memory(const SyncedMemory::size_type size) {
      #if HAS_CUDA
      if (size == 0) {
        device_ptr_ = nullptr;
        return;
      }

      if (size > max_size_device()) {
        throw std::bad_alloc();
      }

      #if DO_FREE_MEMORY_CHECKING
      if (size > available_size_device()) {
        throw std::bad_alloc();
      }
      #endif

      if (cudaMalloc(reinterpret_cast<void**>(&device_ptr_), size_ * sizeof(T)) != cudaSuccess) {
        throw std::bad_alloc();
      }
      #endif
    }

    template<class T>
    void SyncedMemory<T>::allocate_host_memory(const SyncedMemory::size_type size) {
      if (size == 0) {
        host_ptr_ = nullptr;
        return;
      }
      #if HAS_CUDA
      if (cudaMallocHost(reinterpret_cast<void**>(&host_ptr_), size_ * sizeof(T)) != cudaSuccess) {
        throw std::bad_alloc();
      }
      #else
        if (posix_memalign(reinterpret_cast<void**>(&host_ptr_), 64, size * sizeof(T) ) != 0) {
          throw std::bad_alloc();
        }
      #endif
    }

    template<class T>
    inline typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_host_data() {
      copy_to_host();
      return host_ptr_;
    }

    template<class T>
    inline typename SyncedMemory<T>::const_pointer SyncedMemory<T>::const_device_data() {
      copy_to_device();
      return device_ptr_;
    }

    template<class T>
    inline typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_host_data() {
      copy_to_host();
      sync_status_ = SyncStatus::HOST_IS_MUTATED;
      return host_ptr_;
    }

    template<class T>
    inline typename SyncedMemory<T>::pointer SyncedMemory<T>::mutable_device_data() {
      copy_to_device();
      sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
      return device_ptr_;
    }

    template<class T>
    __attribute__((hot))
    void SyncedMemory<T>::copy_to_device() {
      #if HAS_CUDA
      switch(sync_status_) {
        case SyncStatus::UNINITIALIZED:
          allocate_device_memory(size_);
          zero_device();
          sync_status_ = SyncStatus::DEVICE_IS_MUTATED;
          break;
        case SyncStatus::HOST_IS_MUTATED:
          if (device_ptr_ == nullptr) allocate_device_memory(size_);
          #if PRINT_MEMCPY
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
    __attribute__((hot))
    void SyncedMemory<T>::copy_to_host() {
      switch(sync_status_) {
        case SyncStatus::UNINITIALIZED:
          allocate_host_memory(size_);
          zero_host();
          sync_status_ = SyncStatus::HOST_IS_MUTATED;
          break;
        case SyncStatus::DEVICE_IS_MUTATED:
      #if HAS_CUDA
          if (host_ptr_ == nullptr) allocate_host_memory(size_);
          #if PRINT_MEMCPY
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
    inline void SyncedMemory<T>::zero_device() noexcept {
      #if HAS_CUDA
      if (size_ == 0) return;
      assert(device_ptr_);
      #if PRINT_MEMSET
        std::cout << "INFO(SyncedMemory): device zero" << std::endl;
      #endif
      auto status = cudaMemset(device_ptr_, 0, size_ * sizeof(T));
      assert(status == cudaSuccess);
      #endif
    }

    template<class T>
    inline void SyncedMemory<T>::zero_host() noexcept {
      assert(host_ptr_);
      #if PRINT_MEMSET
        std::cout << "INFO(SyncedMemory): host zero" << std::endl;
      #endif
      memset(host_ptr_, 0, size_);
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
    void SyncedMemory<T>::free_device_memory() noexcept {
      #if HAS_CUDA
      if (device_ptr_ != nullptr) {
        auto status = cudaFree(device_ptr_);
        #if ALLOW_GLOBAL_SYNCEDMEMORY
        assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
        #else
        assert(status == cudaSuccess)
        #endif
        device_ptr_ = nullptr;
      }
      #endif
    }

    template<class T>
    typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size() const noexcept {
      #if HAS_CUDA
      return std::min(max_size_host(), max_size_device());
      #else
      return max_size_host();
      #endif
    }

    template<class T>
    void SyncedMemory<T>::resize(SyncedMemory::size_type size) {
      size_ = size;
      free_device_memory();
      free_host_memory();
      sync_status_ = SyncStatus::UNINITIALIZED;
    }

    template<class T>
    typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size_device() const noexcept {
      #if HAS_CUDA
      size_t free;
      size_t total;
      CHECK_CUDA_STATUS(cudaMemGetInfo(&free, &total));
      return total / sizeof(value_type);
      #else
      return 0;
      #endif
    }

    template<class T>
    typename SyncedMemory<T>::size_type SyncedMemory<T>::available_size_device() const noexcept {
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
    void SyncedMemory<T>::free_host_memory() noexcept {
      if (host_ptr_) {
        #if HAS_CUDA
        auto status = cudaFreeHost(host_ptr_);
        #if ALLOW_GLOBAL_SYNCEDMEMORY
        assert(status == cudaSuccess || status == cudaErrorCudartUnloading);
        #else
        assert(status == cudaSuccess)
        #endif
        #else
        free(host_ptr_);
        #endif
        host_ptr_ = nullptr;
      }
    }

    template<class T>
    constexpr typename SyncedMemory<T>::size_type SyncedMemory<T>::max_size_host() const noexcept {
      return std::numeric_limits<size_type>::max() / sizeof(value_type);
    }
}

#endif //JAMS_SYNCED_MEMORY_H
