//
// Created by Codex on 2026-02-16.
//

#ifndef JAMS_RING_STORAGE_H
#define JAMS_RING_STORAGE_H

#include <jams/containers/multiarray.h>

#include <array>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace jams {

/// @brief Ring-buffer storage for N-dimensional trivially-copyable data.
///
/// Supports in-memory storage using jams::MultiArray or file-backed storage via mmap.
/// Ring semantics are applied on a configurable axis.
template<typename T, std::size_t N>
class RingStorage {
public:
  using value_type = T;
  using size_type = std::size_t;
  using shape_type = std::array<size_type, N>;

  static_assert(std::is_trivially_copyable<T>::value, "RingStorage<T,N> requires trivially copyable T");

  explicit RingStorage(const size_type ring_axis = 0)
      : ring_axis_(ring_axis)
  {
    if (ring_axis_ >= N)
    {
      throw std::runtime_error("RingStorage ring_axis out of range");
    }
  }

  RingStorage(const RingStorage&) = delete;
  RingStorage& operator=(const RingStorage&) = delete;
  RingStorage(RingStorage&&) = delete;
  RingStorage& operator=(RingStorage&&) = delete;

  ~RingStorage()
  {
    release_file_backed_();
  }

  void resize(const shape_type& shape, const bool use_file_backed)
  {
    shape_ = shape;
    ring_offset_ = 0;

    const size_type bytes = required_bytes();
    const bool should_use_file_backed = use_file_backed && bytes > 0;

    if (should_use_file_backed)
    {
      in_memory_.clear();
      allocate_file_backed_(bytes);
      std::memset(mapped_data_, 0, mapped_bytes_);
      using_file_backed_ = true;
      return;
    }

    release_file_backed_();
    using_file_backed_ = false;
    in_memory_.resize(shape_);
    in_memory_.zero();
  }

  [[nodiscard]] const shape_type& shape() const
  {
    return shape_;
  }

  [[nodiscard]] size_type size(const size_type dim) const
  {
    assert(dim < N);
    return shape_[dim];
  }

  [[nodiscard]] size_type required_bytes() const
  {
    const auto checked_mul = [](const size_type a, const size_type b) -> size_type
    {
      if (a != 0 && b > std::numeric_limits<size_type>::max() / a)
      {
        throw std::runtime_error("RingStorage size overflow");
      }
      return a * b;
    };

    size_type elements = 1;
    for (const auto dim : shape_)
    {
      elements = checked_mul(elements, dim);
    }
    return checked_mul(elements, sizeof(value_type));
  }

  [[nodiscard]] bool using_file_backed_ring_buffer() const
  {
    return using_file_backed_;
  }

  [[nodiscard]] const std::string& file_path() const
  {
    return mapped_path_;
  }

  void advance_ring_window(const size_type overlap)
  {
    if (shape_[ring_axis_] == 0)
    {
      return;
    }

    assert(overlap < shape_[ring_axis_]);
    ring_offset_ = (ring_offset_ + (shape_[ring_axis_] - overlap)) % shape_[ring_axis_];
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == N)>>
  value_type& operator()(const Args... args)
  {
    return at_(shape_type{static_cast<size_type>(args)...});
  }

  template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) == N)>>
  const value_type& operator()(const Args... args) const
  {
    return at_(shape_type{static_cast<size_type>(args)...});
  }

  value_type& operator()(const shape_type& idx)
  {
    return at_(idx);
  }

  const value_type& operator()(const shape_type& idx) const
  {
    return at_(idx);
  }

private:
  size_type map_ring_index_(const size_type logical_index) const
  {
    assert(logical_index < shape_[ring_axis_]);
    if (shape_[ring_axis_] == 0)
    {
      return 0;
    }
    return (ring_offset_ + logical_index) % shape_[ring_axis_];
  }

  shape_type map_index_(const shape_type& logical_idx) const
  {
    shape_type mapped = logical_idx;
    mapped[ring_axis_] = map_ring_index_(logical_idx[ring_axis_]);
    return mapped;
  }

  size_type flat_index_(const shape_type& mapped_idx) const
  {
    size_type flat = 0;
    for (size_type i = 0; i < N; ++i)
    {
      assert(mapped_idx[i] < shape_[i]);
      flat = flat * shape_[i] + mapped_idx[i];
    }
    return flat;
  }

  value_type& at_(const shape_type& logical_idx)
  {
    const auto mapped_idx = map_index_(logical_idx);
    if (using_file_backed_)
    {
      return mapped_data_[flat_index_(mapped_idx)];
    }
    return in_memory_(mapped_idx);
  }

  const value_type& at_(const shape_type& logical_idx) const
  {
    const auto mapped_idx = map_index_(logical_idx);
    if (using_file_backed_)
    {
      return mapped_data_[flat_index_(mapped_idx)];
    }
    return in_memory_(mapped_idx);
  }

  void allocate_file_backed_(const size_type bytes)
  {
    release_file_backed_();

    const std::filesystem::path file_template =
        std::filesystem::temp_directory_path() / "jams_ring_storage_XXXXXX";
    std::string file_path = file_template.string();
    file_path.push_back('\0');

    mapped_fd_ = mkstemp(file_path.data());
    if (mapped_fd_ == -1)
    {
      throw std::runtime_error("Failed to create temporary file for RingStorage");
    }

    mapped_path_ = std::string(file_path.c_str());

    if (ftruncate(mapped_fd_, static_cast<off_t>(bytes)) != 0)
    {
      const auto saved_errno = errno;
      close(mapped_fd_);
      mapped_fd_ = -1;
      std::error_code ec;
      std::filesystem::remove(mapped_path_, ec);
      mapped_path_.clear();
      throw std::runtime_error("Failed to resize RingStorage file: " + std::string(std::strerror(saved_errno)));
    }

    void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapped_fd_, 0);
    if (ptr == MAP_FAILED)
    {
      const auto saved_errno = errno;
      close(mapped_fd_);
      mapped_fd_ = -1;
      std::error_code ec;
      std::filesystem::remove(mapped_path_, ec);
      mapped_path_.clear();
      throw std::runtime_error("Failed to map RingStorage file: " + std::string(std::strerror(saved_errno)));
    }

    mapped_data_ = static_cast<value_type*>(ptr);
    mapped_bytes_ = bytes;
  }

  void release_file_backed_()
  {
    if (mapped_data_)
    {
      munmap(mapped_data_, mapped_bytes_);
      mapped_data_ = nullptr;
    }
    mapped_bytes_ = 0;

    if (mapped_fd_ != -1)
    {
      close(mapped_fd_);
      mapped_fd_ = -1;
    }

    if (!mapped_path_.empty())
    {
      std::error_code ec;
      std::filesystem::remove(mapped_path_, ec);
      mapped_path_.clear();
    }
  }

  size_type ring_axis_ = 0;
  shape_type shape_ = {0};
  size_type ring_offset_ = 0;
  bool using_file_backed_ = false;

  jams::MultiArray<value_type, N> in_memory_;

  value_type* mapped_data_ = nullptr;
  size_type mapped_bytes_ = 0;
  int mapped_fd_ = -1;
  std::string mapped_path_;
};

} // namespace jams

#endif // JAMS_RING_STORAGE_H
