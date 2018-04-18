//
// Created by Joe Barker on 2017/11/30.
//

#include <cuda_runtime.h>

dim3 cuda_grid_size(const dim3 &block_size, const dim3 &grid_size) {
  return {(grid_size.x + block_size.x - 1) / block_size.x,
  (grid_size.y + block_size.y - 1) / block_size.y,
  (grid_size.z + block_size.z - 1) / block_size.z};
}
