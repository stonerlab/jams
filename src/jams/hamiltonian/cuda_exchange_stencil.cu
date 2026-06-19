// Copyright 2026 Joseph Barker. All rights reserved.

#include <jams/hamiltonian/exchange_backend.h>

#if HAS_CUDA

#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <jams/core/globals.h>
#include <jams/cuda/cuda_array_kernels.h>
#include <jams/cuda/cuda_common.h>

namespace {

template <int Storage>
__device__ __forceinline__ constexpr int tensor_component_count() {
  if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Isotropic)) {
    return 1;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Anisotropic)) {
    return 3;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Symmetric)) {
    return 6;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)) {
    return 3;
  } else {
    return 9;
  }
}

template <int Storage>
__device__ __forceinline__ void accumulate_tensor_field(
    const jams::Real* __restrict__ values,
    const jams::Real sx,
    const jams::Real sy,
    const jams::Real sz,
    jams::Real& hx,
    jams::Real& hy,
    jams::Real& hz) {
  if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Isotropic)) {
    const jams::Real j0 = values[0];
    hx += j0 * sx;
    hy += j0 * sy;
    hz += j0 * sz;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Anisotropic)) {
    hx += values[0] * sx;
    hy += values[1] * sy;
    hz += values[2] * sz;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Symmetric)) {
    hx += values[0] * sx + values[1] * sy + values[2] * sz;
    hy += values[1] * sx + values[3] * sy + values[4] * sz;
    hz += values[2] * sx + values[4] * sy + values[5] * sz;
  } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)) {
    hx += values[0] * sy + values[1] * sz;
    hy += -values[0] * sx + values[2] * sz;
    hz += -values[1] * sx - values[2] * sy;
  } else {
    hx += values[0] * sx + values[1] * sy + values[2] * sz;
    hy += values[3] * sx + values[4] * sy + values[5] * sz;
    hz += values[6] * sx + values[7] * sy + values[8] * sz;
  }
}

int apply_host_stencil_boundary(
    const int value,
    const int size,
    const bool periodic) {
  if (!periodic && (value < 0 || value >= size)) {
    return -1;
  }
  int wrapped = value % size;
  if (wrapped < 0) {
    wrapped += size;
  }
  return wrapped;
}

template <int Storage, bool FullyPeriodic>
__global__ void cuda_exchange_stencil_mapped_grouped_field_kernel(
    const int num_spins,
    const int num_basis_sites,
    const int num_cells,
    const int* __restrict__ group_offsets,
    const int* __restrict__ group_entry_offsets,
    const int* __restrict__ group_translation_ids,
    const int* __restrict__ translation_target_cells,
    const int* __restrict__ target_basis,
    const jams::Real* __restrict__ values,
    const jams::Real* __restrict__ spins,
    jams::Real* __restrict__ field) {
  const int site = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (site >= num_spins) {
    return;
  }

  const int basis = site % num_basis_sites;
  const int cell = site / num_basis_sites;

  jams::Real hx = 0;
  jams::Real hy = 0;
  jams::Real hz = 0;

  for (int group = group_offsets[basis]; group < group_offsets[basis + 1]; ++group) {
    const int target_cell = translation_target_cells[group_translation_ids[group] * num_cells + cell];
    if constexpr (!FullyPeriodic) {
      if (target_cell < 0) {
        continue;
      }
    }

    for (int entry = group_entry_offsets[group]; entry < group_entry_offsets[group + 1]; ++entry) {
      const int target_site = target_cell * num_basis_sites + target_basis[entry];
      const int spin_base = 3 * target_site;
      const jams::Real sx = spins[spin_base + 0];
      const jams::Real sy = spins[spin_base + 1];
      const jams::Real sz = spins[spin_base + 2];
      accumulate_tensor_field<Storage>(
          values + tensor_component_count<Storage>() * entry,
          sx,
          sy,
          sz,
          hx,
          hy,
          hz);
    }
  }

  const int field_base = 3 * site;
  field[field_base + 0] = hx;
  field[field_base + 1] = hy;
  field[field_base + 2] = hz;
}

__device__ __forceinline__ int wrap_periodic_axis_fast(
    int value,
    const int size) {
  if (value < 0) {
    value += size;
  } else if (value >= size) {
    value -= size;
  }

  if (value < 0 || value >= size) {
    value %= size;
    if (value < 0) {
      value += size;
    }
  }
  return value;
}

template <int Storage>
__global__ void cuda_exchange_stencil_periodic_direct_flat_field_kernel(
    const int num_spins,
    const int nx,
    const int ny,
    const int nz,
    const int num_basis_sites,
    const int* __restrict__ entry_offsets,
    const int* __restrict__ entry_dx,
    const int* __restrict__ entry_dy,
    const int* __restrict__ entry_dz,
    const int* __restrict__ target_basis,
    const jams::Real* __restrict__ values,
    const jams::Real* __restrict__ spins,
    jams::Real* __restrict__ field) {
  const int site = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (site >= num_spins) {
    return;
  }

  const int basis = site % num_basis_sites;
  int cell = site / num_basis_sites;
  const int cell_z = cell % nz;
  cell /= nz;
  const int cell_y = cell % ny;
  const int cell_x = cell / ny;

  jams::Real hx = 0;
  jams::Real hy = 0;
  jams::Real hz = 0;

  for (int entry = entry_offsets[basis]; entry < entry_offsets[basis + 1]; ++entry) {
    const int target_x = wrap_periodic_axis_fast(cell_x + entry_dx[entry], nx);
    const int target_y = wrap_periodic_axis_fast(cell_y + entry_dy[entry], ny);
    const int target_z = wrap_periodic_axis_fast(cell_z + entry_dz[entry], nz);
    const int target_cell = (target_x * ny + target_y) * nz + target_z;
    const int target_site = target_cell * num_basis_sites + target_basis[entry];
    const int spin_base = 3 * target_site;
    const jams::Real sx = spins[spin_base + 0];
    const jams::Real sy = spins[spin_base + 1];
    const jams::Real sz = spins[spin_base + 2];
    accumulate_tensor_field<Storage>(
        values + tensor_component_count<Storage>() * entry,
        sx,
        sy,
        sz,
        hx,
        hy,
        hz);
  }

  const int field_base = 3 * site;
  field[field_base + 0] = hx;
  field[field_base + 1] = hy;
  field[field_base + 2] = hz;
}

template <int Storage>
void launch_cuda_exchange_stencil_field_kernel(
    const bool use_direct_periodic_kernel,
    const bool fully_periodic,
    const dim3 grid_size,
    const dim3 block_size,
    const int num_spins,
    const int nx,
    const int ny,
    const int nz,
    const int num_basis_sites,
    const int num_cells,
    const int* entry_offsets,
    const int* entry_dx,
    const int* entry_dy,
    const int* entry_dz,
    const int* group_offsets,
    const int* group_entry_offsets,
    const int* group_translation_ids,
    const int* group_dx,
    const int* group_dy,
    const int* group_dz,
    const int* translation_target_cells,
    const int* target_basis,
    const jams::Real* values,
    const jams::Real* spins,
    jams::Real* field,
    cudaStream_t stream) {
  if (use_direct_periodic_kernel) {
    cuda_exchange_stencil_periodic_direct_flat_field_kernel<Storage>
        <<<grid_size, block_size, 0, stream>>>(
            num_spins,
            nx,
            ny,
            nz,
            num_basis_sites,
            entry_offsets,
            entry_dx,
            entry_dy,
            entry_dz,
            target_basis,
            values,
            spins,
            field);
  } else if (fully_periodic) {
    cuda_exchange_stencil_mapped_grouped_field_kernel<Storage, true>
        <<<grid_size, block_size, 0, stream>>>(
            num_spins,
            num_basis_sites,
            num_cells,
            group_offsets,
            group_entry_offsets,
            group_translation_ids,
            translation_target_cells,
            target_basis,
            values,
            spins,
            field);
  } else {
    cuda_exchange_stencil_mapped_grouped_field_kernel<Storage, false>
        <<<grid_size, block_size, 0, stream>>>(
            num_spins,
            num_basis_sites,
            num_cells,
            group_offsets,
            group_entry_offsets,
            group_translation_ids,
            translation_target_cells,
            target_basis,
            values,
            spins,
            field);
  }
}

}  // namespace

class CudaExchangeStencilBackend : public ExchangeStencilBackend {
public:
  CudaExchangeStencilBackend(
      const libconfig::Setting& settings,
      std::shared_ptr<const ExchangeInteractionSetup> setup,
      cudaStream_t stream);

  [[nodiscard]] bool uses_device() const override {
    return true;
  }

  [[nodiscard]] bool supports_calculate_fields_in_parallel() const override {
    return false;
  }

  void calculate_fields(jams::Real time, const SpinArray& spins, FieldArray& field) override;
  void calculate_energies(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override;
  [[nodiscard]] jams::Real calculate_total_energy(
      jams::Real time,
      const SpinArray& spins,
      FieldArray& field,
      EnergyArray& energy) override;

private:
  void upload_stencil_to_device();

  cudaStream_t stream_ = nullptr;
  jams::MultiArray<int, 1> device_entry_offsets_;
  jams::MultiArray<int, 1> device_entry_dx_;
  jams::MultiArray<int, 1> device_entry_dy_;
  jams::MultiArray<int, 1> device_entry_dz_;
  jams::MultiArray<int, 1> device_target_basis_;
  jams::MultiArray<int, 1> device_group_offsets_;
  jams::MultiArray<int, 1> device_group_entry_offsets_;
  jams::MultiArray<int, 1> device_group_translation_ids_;
  jams::MultiArray<int, 1> device_group_dx_;
  jams::MultiArray<int, 1> device_group_dy_;
  jams::MultiArray<int, 1> device_group_dz_;
  jams::MultiArray<int, 1> device_translation_target_cells_;
  jams::MultiArray<jams::Real, 1> device_values_;
  int num_cell_translations_ = 0;
  int num_stencil_entries_ = 0;
  int num_stencil_groups_ = 0;
  int field_kernel_block_size_ = 128;
  int components_per_entry_ = 1;
  bool use_direct_periodic_kernel_ = false;
  bool fully_periodic_ = false;
  jams::InteractionTensorStorage tensor_storage_ = jams::InteractionTensorStorage::Isotropic;
};

CudaExchangeStencilBackend::CudaExchangeStencilBackend(
    const libconfig::Setting& settings,
    std::shared_ptr<const ExchangeInteractionSetup> setup,
    cudaStream_t stream)
    : ExchangeStencilBackend(settings, std::move(setup)),
      stream_(stream) {
  upload_stencil_to_device();
}

void CudaExchangeStencilBackend::upload_stencil_to_device() {
  struct UploadEntry {
    int target_basis = 0;
    int translation_id = 0;
    jams::Vec<int, 3> translation = {0, 0, 0};
    jams::Mat<jams::Real, 3, 3> tensor = kZeroMat3R;
  };

  const auto& entries_by_basis = stencil_entries_by_basis();
  num_stencil_entries_ = 0;
  tensor_storage_ = jams::InteractionTensorStorage::Auto;
  for (const auto& entries : entries_by_basis) {
    num_stencil_entries_ += static_cast<int>(entries.size());
    for (const auto& entry : entries) {
      const auto entry_storage = jams::detail::classify_tensor_storage(entry.tensor, 0.0);
      tensor_storage_ = tensor_storage_ == jams::InteractionTensorStorage::Auto
          ? entry_storage
          : jams::detail::combine_tensor_storage(tensor_storage_, entry_storage);
    }
  }
  if (tensor_storage_ == jams::InteractionTensorStorage::Auto) {
    tensor_storage_ = jams::InteractionTensorStorage::Isotropic;
  }
  components_per_entry_ = jams::interaction_tensor_storage_components(tensor_storage_);

  std::map<std::tuple<int, int, int>, int> translation_ids;
  std::vector<jams::Vec<int, 3>> translations;
  const auto get_translation_id = [&](const jams::Vec<int, 3>& translation) {
    const auto key = std::make_tuple(translation[0], translation[1], translation[2]);
    const auto found = translation_ids.find(key);
    if (found != translation_ids.end()) {
      return found->second;
    }

    const auto id = static_cast<int>(translations.size());
    translation_ids.emplace(key, id);
    translations.push_back(translation);
    return id;
  };

  std::vector<int> host_entry_offsets(num_basis_sites() + 1, 0);
  std::vector<int> host_group_offsets(num_basis_sites() + 1, 0);
  std::vector<int> host_group_entry_offsets;
  std::vector<int> host_group_translation_ids;
  std::vector<int> host_group_dx;
  std::vector<int> host_group_dy;
  std::vector<int> host_group_dz;
  std::vector<UploadEntry> flat_entries;
  flat_entries.reserve(num_stencil_entries_);

  for (auto basis = 0; basis < num_basis_sites(); ++basis) {
    host_entry_offsets[basis] = static_cast<int>(flat_entries.size());
    host_group_offsets[basis] = static_cast<int>(host_group_translation_ids.size());

    std::map<int, std::vector<UploadEntry>> grouped_entries;
    for (const auto& entry : entries_by_basis[basis]) {
      const int translation_id = get_translation_id(entry.lattice_translation);
      grouped_entries[translation_id].push_back(UploadEntry{
          entry.basis_site_j,
          translation_id,
          entry.lattice_translation,
          entry.tensor});
    }

    for (const auto& [translation_id, entries] : grouped_entries) {
      const auto& translation = translations[translation_id];
      host_group_entry_offsets.push_back(static_cast<int>(flat_entries.size()));
      host_group_translation_ids.push_back(translation_id);
      host_group_dx.push_back(translation[0]);
      host_group_dy.push_back(translation[1]);
      host_group_dz.push_back(translation[2]);
      flat_entries.insert(flat_entries.end(), entries.begin(), entries.end());
    }
  }
  host_entry_offsets[num_basis_sites()] = static_cast<int>(flat_entries.size());
  host_group_offsets[num_basis_sites()] = static_cast<int>(host_group_translation_ids.size());
  host_group_entry_offsets.push_back(static_cast<int>(flat_entries.size()));

  num_cell_translations_ = static_cast<int>(translations.size());
  num_stencil_groups_ = static_cast<int>(host_group_translation_ids.size());

  const auto size = lattice_size();
  const auto periodic = periodic_boundaries();
  const int num_cells = size[0] * size[1] * size[2];
  fully_periodic_ = periodic[0] && periodic[1] && periodic[2];

  // Direct periodic wrapping wins when translations are mostly unreused. The
  // mapped path wins when many stencil entries share a small set of cell shifts.
  use_direct_periodic_kernel_ = fully_periodic_ && num_stencil_entries_ <= 2 * num_stencil_groups_;
  field_kernel_block_size_ = use_direct_periodic_kernel_ && globals::num_spins >= 65536 ? 256 : 128;

  device_entry_offsets_.resize(host_entry_offsets.size());
  if (use_direct_periodic_kernel_) {
    device_entry_dx_.resize(flat_entries.size());
    device_entry_dy_.resize(flat_entries.size());
    device_entry_dz_.resize(flat_entries.size());
  }
  device_target_basis_.resize(flat_entries.size());
  device_group_offsets_.resize(host_group_offsets.size());
  device_group_entry_offsets_.resize(host_group_entry_offsets.size());
  device_group_translation_ids_.resize(host_group_translation_ids.size());
  device_group_dx_.resize(host_group_dx.size());
  device_group_dy_.resize(host_group_dy.size());
  device_group_dz_.resize(host_group_dz.size());
  device_values_.resize(flat_entries.size() * components_per_entry_);

  for (std::size_t i = 0; i < host_entry_offsets.size(); ++i) {
    device_entry_offsets_(i) = host_entry_offsets[i];
  }
  for (std::size_t i = 0; i < host_group_offsets.size(); ++i) {
    device_group_offsets_(i) = host_group_offsets[i];
  }
  for (std::size_t i = 0; i < host_group_entry_offsets.size(); ++i) {
    device_group_entry_offsets_(i) = host_group_entry_offsets[i];
  }
  for (std::size_t i = 0; i < host_group_translation_ids.size(); ++i) {
    device_group_translation_ids_(i) = host_group_translation_ids[i];
    device_group_dx_(i) = host_group_dx[i];
    device_group_dy_(i) = host_group_dy[i];
    device_group_dz_(i) = host_group_dz[i];
  }

  std::vector<jams::Real> packed_components;
  packed_components.reserve(9);
  for (std::size_t entry_index = 0; entry_index < flat_entries.size(); ++entry_index) {
    device_target_basis_(entry_index) = flat_entries[entry_index].target_basis;
    if (use_direct_periodic_kernel_) {
      device_entry_dx_(entry_index) = flat_entries[entry_index].translation[0];
      device_entry_dy_(entry_index) = flat_entries[entry_index].translation[1];
      device_entry_dz_(entry_index) = flat_entries[entry_index].translation[2];
    }

    packed_components.clear();
    jams::detail::pack_tensor_components(
        packed_components,
        tensor_storage_,
        flat_entries[entry_index].tensor,
        0.0);
    if (static_cast<int>(packed_components.size()) != components_per_entry_) {
      throw std::runtime_error("internal error packing exchange-stencil tensor components");
    }
    for (auto component = 0; component < components_per_entry_; ++component) {
      device_values_(components_per_entry_ * entry_index + component) = packed_components[component];
    }
  }

  if (!use_direct_periodic_kernel_) {
    device_translation_target_cells_.resize(num_cell_translations_ * num_cells);

    for (auto translation_id = 0; translation_id < num_cell_translations_; ++translation_id) {
      const auto& translation = translations[translation_id];
      for (auto cell_x = 0; cell_x < size[0]; ++cell_x) {
        for (auto cell_y = 0; cell_y < size[1]; ++cell_y) {
          for (auto cell_z = 0; cell_z < size[2]; ++cell_z) {
            const int source_cell = (cell_x * size[1] + cell_y) * size[2] + cell_z;
            const int target_x = apply_host_stencil_boundary(
                cell_x + translation[0],
                size[0],
                periodic[0]);
            const int target_y = apply_host_stencil_boundary(
                cell_y + translation[1],
                size[1],
                periodic[1]);
            const int target_z = apply_host_stencil_boundary(
                cell_z + translation[2],
                size[2],
                periodic[2]);
            int target_cell = -1;
            if (target_x >= 0 && target_y >= 0 && target_z >= 0) {
              target_cell = (target_x * size[1] + target_y) * size[2] + target_z;
            }
            device_translation_target_cells_(translation_id * num_cells + source_cell) = target_cell;
          }
        }
      }
    }
  }

  std::cout << "    cuda stencil tensor storage: "
            << jams::to_string(tensor_storage_) << "\n";
  std::cout << "    cuda stencil cell translations: " << num_cell_translations_ << "\n";
  std::cout << "    cuda stencil grouped translations: " << num_stencil_groups_ << "\n";
  std::cout << "    cuda stencil field kernel: "
            << (use_direct_periodic_kernel_ ? "direct-periodic" : "mapped") << "\n";
  std::cout << "    cuda stencil field kernel block size: " << field_kernel_block_size_ << "\n";

  // Ensure device copies are materialized before the first field kernel.
  if (use_direct_periodic_kernel_) {
    (void)device_entry_offsets_.device_data();
    (void)device_entry_dx_.device_data();
    (void)device_entry_dy_.device_data();
    (void)device_entry_dz_.device_data();
  } else {
    (void)device_group_offsets_.device_data();
    (void)device_group_entry_offsets_.device_data();
    (void)device_group_translation_ids_.device_data();
    (void)device_group_dx_.device_data();
    (void)device_group_dy_.device_data();
    (void)device_group_dz_.device_data();
    (void)device_translation_target_cells_.device_data();
  }
  (void)device_target_basis_.device_data();
  (void)device_values_.device_data();
}

void CudaExchangeStencilBackend::calculate_fields(jams::Real time, const SpinArray& spins, FieldArray& field) {
  const dim3 block_size = {static_cast<unsigned int>(field_kernel_block_size_), 1, 1};
  const dim3 grid_size = cuda_grid_size(
      block_size,
      {static_cast<unsigned int>(globals::num_spins), 1, 1});
  const auto size = lattice_size();
  const int num_cells = size[0] * size[1] * size[2];
  const int* translation_target_cells = use_direct_periodic_kernel_
      ? nullptr
      : device_translation_target_cells_.device_data();
  const int* entry_dx = use_direct_periodic_kernel_ ? device_entry_dx_.device_data() : nullptr;
  const int* entry_dy = use_direct_periodic_kernel_ ? device_entry_dy_.device_data() : nullptr;
  const int* entry_dz = use_direct_periodic_kernel_ ? device_entry_dz_.device_data() : nullptr;
  const int* entry_offsets = use_direct_periodic_kernel_ ? device_entry_offsets_.device_data() : nullptr;
  const int* group_offsets = use_direct_periodic_kernel_ ? nullptr : device_group_offsets_.device_data();
  const int* group_entry_offsets = use_direct_periodic_kernel_
      ? nullptr
      : device_group_entry_offsets_.device_data();
  const int* group_translation_ids = use_direct_periodic_kernel_
      ? nullptr
      : device_group_translation_ids_.device_data();
  const int* group_dx = use_direct_periodic_kernel_ ? nullptr : device_group_dx_.device_data();
  const int* group_dy = use_direct_periodic_kernel_ ? nullptr : device_group_dy_.device_data();
  const int* group_dz = use_direct_periodic_kernel_ ? nullptr : device_group_dz_.device_data();

  switch (tensor_storage_) {
    case jams::InteractionTensorStorage::Isotropic:
      launch_cuda_exchange_stencil_field_kernel<
          static_cast<int>(jams::InteractionTensorStorage::Isotropic)>(
          use_direct_periodic_kernel_,
          fully_periodic_,
          grid_size,
          block_size,
          globals::num_spins,
          size[0],
          size[1],
          size[2],
          num_basis_sites(),
          num_cells,
          entry_offsets,
          entry_dx,
          entry_dy,
          entry_dz,
          group_offsets,
          group_entry_offsets,
          group_translation_ids,
          group_dx,
          group_dy,
          group_dz,
          translation_target_cells,
          device_target_basis_.device_data(),
          device_values_.device_data(),
          spins.device_data(),
          field.mutable_device_data(),
          stream_);
      break;
    case jams::InteractionTensorStorage::Anisotropic:
      launch_cuda_exchange_stencil_field_kernel<
          static_cast<int>(jams::InteractionTensorStorage::Anisotropic)>(
          use_direct_periodic_kernel_,
          fully_periodic_,
          grid_size,
          block_size,
          globals::num_spins,
          size[0],
          size[1],
          size[2],
          num_basis_sites(),
          num_cells,
          entry_offsets,
          entry_dx,
          entry_dy,
          entry_dz,
          group_offsets,
          group_entry_offsets,
          group_translation_ids,
          group_dx,
          group_dy,
          group_dz,
          translation_target_cells,
          device_target_basis_.device_data(),
          device_values_.device_data(),
          spins.device_data(),
          field.mutable_device_data(),
          stream_);
      break;
    case jams::InteractionTensorStorage::Symmetric:
      launch_cuda_exchange_stencil_field_kernel<
          static_cast<int>(jams::InteractionTensorStorage::Symmetric)>(
          use_direct_periodic_kernel_,
          fully_periodic_,
          grid_size,
          block_size,
          globals::num_spins,
          size[0],
          size[1],
          size[2],
          num_basis_sites(),
          num_cells,
          entry_offsets,
          entry_dx,
          entry_dy,
          entry_dz,
          group_offsets,
          group_entry_offsets,
          group_translation_ids,
          group_dx,
          group_dy,
          group_dz,
          translation_target_cells,
          device_target_basis_.device_data(),
          device_values_.device_data(),
          spins.device_data(),
          field.mutable_device_data(),
          stream_);
      break;
    case jams::InteractionTensorStorage::Antisymmetric:
      launch_cuda_exchange_stencil_field_kernel<
          static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)>(
          use_direct_periodic_kernel_,
          fully_periodic_,
          grid_size,
          block_size,
          globals::num_spins,
          size[0],
          size[1],
          size[2],
          num_basis_sites(),
          num_cells,
          entry_offsets,
          entry_dx,
          entry_dy,
          entry_dz,
          group_offsets,
          group_entry_offsets,
          group_translation_ids,
          group_dx,
          group_dy,
          group_dz,
          translation_target_cells,
          device_target_basis_.device_data(),
          device_values_.device_data(),
          spins.device_data(),
          field.mutable_device_data(),
          stream_);
      break;
    case jams::InteractionTensorStorage::General:
      launch_cuda_exchange_stencil_field_kernel<
          static_cast<int>(jams::InteractionTensorStorage::General)>(
          use_direct_periodic_kernel_,
          fully_periodic_,
          grid_size,
          block_size,
          globals::num_spins,
          size[0],
          size[1],
          size[2],
          num_basis_sites(),
          num_cells,
          entry_offsets,
          entry_dx,
          entry_dy,
          entry_dz,
          group_offsets,
          group_entry_offsets,
          group_translation_ids,
          group_dx,
          group_dy,
          group_dz,
          translation_target_cells,
          device_target_basis_.device_data(),
          device_values_.device_data(),
          spins.device_data(),
          field.mutable_device_data(),
          stream_);
      break;
    case jams::InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot launch exchange-stencil field kernel with auto tensor storage");
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void CudaExchangeStencilBackend::calculate_energies(
    jams::Real time,
    const SpinArray& spins,
    FieldArray& field,
    EnergyArray& energy) {
  calculate_fields(time, spins, field);
  cuda_array_dot_product(
      globals::num_spins,
      static_cast<jams::Real>(-0.5),
      spins.device_data(),
      field.device_data(),
      energy.mutable_device_data(),
      stream_);
}

jams::Real CudaExchangeStencilBackend::calculate_total_energy(
    jams::Real time,
    const SpinArray& spins,
    FieldArray& field,
    EnergyArray& energy) {
  calculate_energies(time, spins, field, energy);
  return cuda_reduce_array(energy.device_data(), globals::num_spins, stream_);
}

std::unique_ptr<ExchangeBackendImpl> make_cuda_exchange_stencil_backend(
    const libconfig::Setting& settings,
    std::shared_ptr<const ExchangeInteractionSetup> setup,
    cudaStream_t stream) {
  return std::make_unique<CudaExchangeStencilBackend>(settings, std::move(setup), stream);
}

#endif  // HAS_CUDA
