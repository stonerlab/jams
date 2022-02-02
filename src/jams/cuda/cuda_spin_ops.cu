#include <jams/cuda/cuda_spin_ops.h>
#include <jams/helpers/array_ops.h>

#include <cuda.h>

__device__ void warp_reduce(volatile double* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void
vector_field_key_reduce_kernel(const int *g_ikeys, const double *g_idata, double *g_odata, unsigned int n)
{
  extern __shared__ double3 sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  double3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = g_idata[3*g_ikeys[i]+0];
    mySum.y = g_idata[3*g_ikeys[i]+1];
    mySum.z = g_idata[3*g_ikeys[i]+2];
  }

  if (i + blockSize < n) {
    mySum.x += g_idata[3*g_ikeys[i + blockSize]+0];
    mySum.y += g_idata[3*g_ikeys[i + blockSize]+1];
    mySum.z += g_idata[3*g_ikeys[i + blockSize]+2];
  }

  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 256].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 256].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 256].z;
  }

  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 128].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 128].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 128].z;
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid <  64))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 64].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 64].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 64].z;
  }

  __syncthreads();

  if ( tid < 32 ) {
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) {
        mySum.x += sdata[tid + 32].x;
        mySum.y += sdata[tid + 32].y;
        mySum.z += sdata[tid + 32].z;
      }
      // Reduce final warp using shuffle
      for (int offset = warpSize/2; offset > 0; offset /= 2)
      {
        // Mask 0xFFFFFFFF see: https://stackoverflow.com/questions/50639194/shfl-down-and-shfl-down-sync-give-different-results
        mySum.x += __shfl_down_sync(0xFFFFFFFF, mySum.x, offset);
        mySum.y += __shfl_down_sync(0xFFFFFFFF, mySum.y, offset);
        mySum.z += __shfl_down_sync(0xFFFFFFFF, mySum.z, offset);
      }
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[3 * blockIdx.x + 0] = mySum.x;
    g_odata[3 * blockIdx.x + 1] = mySum.y;
    g_odata[3 * blockIdx.x + 2] = mySum.z;
  }
}


template <unsigned int blockSize>
__global__ void
vector_field_key_multiply_and_reduce_kernel(const int *g_ikeys, const double *g_idata, const double *g_ifactors, double *g_odata, unsigned int n)
{
  extern __shared__ double3 sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  double3 mySum = {0, 0, 0};
  if (i < n) {
    mySum.x = g_idata[3*g_ikeys[i]+0] * g_ifactors[g_ikeys[i]];
    mySum.y = g_idata[3*g_ikeys[i]+1] * g_ifactors[g_ikeys[i]];;
    mySum.z = g_idata[3*g_ikeys[i]+2] * g_ifactors[g_ikeys[i]];;
  }

  if (i + blockSize < n) {
    mySum.x += g_idata[3*g_ikeys[i + blockSize]+0] * g_ifactors[g_ikeys[i + blockSize]];
    mySum.y += g_idata[3*g_ikeys[i + blockSize]+1] * g_ifactors[g_ikeys[i + blockSize]];
    mySum.z += g_idata[3*g_ikeys[i + blockSize]+2] * g_ifactors[g_ikeys[i + blockSize]];
  }

  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 256].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 256].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 256].z;
  }

  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 128].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 128].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 128].z;
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid <  64))
  {
    sdata[tid].x = mySum.x = mySum.x + sdata[tid + 64].x;
    sdata[tid].y = mySum.y = mySum.y + sdata[tid + 64].y;
    sdata[tid].z = mySum.z = mySum.z + sdata[tid + 64].z;
  }

  __syncthreads();

  if ( tid < 32 ) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) {
      mySum.x += sdata[tid + 32].x;
      mySum.y += sdata[tid + 32].y;
      mySum.z += sdata[tid + 32].z;
    }
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
      // Mask 0xFFFFFFFF see: https://stackoverflow.com/questions/50639194/shfl-down-and-shfl-down-sync-give-different-results
      mySum.x += __shfl_down_sync(0xFFFFFFFF, mySum.x, offset);
      mySum.y += __shfl_down_sync(0xFFFFFFFF, mySum.y, offset);
      mySum.z += __shfl_down_sync(0xFFFFFFFF, mySum.z, offset);
    }
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[3 * blockIdx.x + 0] = mySum.x;
    g_odata[3 * blockIdx.x + 1] = mySum.y;
    g_odata[3 * blockIdx.x + 2] = mySum.z;
  }
}



void vector_field_key_reduce(int size, int threads, int blocks,
                                    const int *d_ikeys,
                                    const double *d_idata,
                                    double *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double3) : threads * sizeof(double3);

  switch (threads)
  {
    case 512:
      vector_field_key_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case 256:
      vector_field_key_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case 128:
      vector_field_key_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case 64:
      vector_field_key_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case 32:
      vector_field_key_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case 16:
      vector_field_key_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case  8:
      vector_field_key_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case  4:
      vector_field_key_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case  2:
      vector_field_key_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;

    case  1:
      vector_field_key_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_odata, size);
      break;
  }

}

void vector_field_key_multiply_and_reduce(int size, int threads, int blocks,
                             const int *d_ikeys,
                             const double *d_idata,
                             const double *d_ifactors,
                             double *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(double3) : threads * sizeof(double3);

  switch (threads)
  {
    case 512:
      vector_field_key_multiply_and_reduce_kernel<512><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case 256:
      vector_field_key_multiply_and_reduce_kernel<256><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case 128:
      vector_field_key_multiply_and_reduce_kernel<128><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case 64:
      vector_field_key_multiply_and_reduce_kernel< 64><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case 32:
      vector_field_key_multiply_and_reduce_kernel< 32><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case 16:
      vector_field_key_multiply_and_reduce_kernel< 16><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case  8:
      vector_field_key_multiply_and_reduce_kernel<  8><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case  4:
      vector_field_key_multiply_and_reduce_kernel<  4><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case  2:
      vector_field_key_multiply_and_reduce_kernel<  2><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;

    case  1:
      vector_field_key_multiply_and_reduce_kernel<  1><<< dimGrid, dimBlock, smemSize >>>(d_ikeys, d_idata, d_ifactors, d_odata, size);
      break;
  }

}

unsigned int next_pow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

Vec3 jams::cuda_sum_spins(const jams::MultiArray<double, 2> &spins,
                          const jams::MultiArray<int, 1> &indices) {

  // Use kernel to sum sx, sy, sz within thread blocks into an array
  static jams::MultiArray<double, 2> odata;


  int max_threads = 256;
  int size = indices.size();
  int threads = (size < max_threads*2) ? next_pow2((size + 1)/ 2) : max_threads;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  odata.resize(blocks, 3);

  vector_field_key_reduce(size, threads, blocks, indices.device_data(), spins.device_data(), odata.device_data());

  return jams::reduce_vector_field(odata);
}


Vec3 jams::cuda_sum_spins_moments(const jams::MultiArray<double, 2> &spins,
                          const jams::MultiArray<double, 1>& moments,
                          const jams::MultiArray<int, 1> &indices) {

  // Use kernel to sum sx, sy, sz within thread blocks into an array
  static jams::MultiArray<double, 2> odata;


  int max_threads = 256;
  int size = indices.size();
  int threads = (size < max_threads*2) ? next_pow2((size + 1)/ 2) : max_threads;
  int blocks = (size + (threads * 2 - 1)) / (threads * 2);

  odata.resize(blocks, 3);

  vector_field_key_multiply_and_reduce(size, threads, blocks, indices.device_data(), spins.device_data(), moments.device_data(), odata.device_data());

  return jams::reduce_vector_field(odata);
}



__global__ void cuda_rotate_spins_kernel(double* spins, const int* indices, const unsigned size,
                                         double Rxx, double Rxy, double Rxz,
                                         double Ryx, double Ryy, double Ryz,
                                         double Rzx, double Rzy, double Rzz) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    double s[3] = {spins[3*indices[idx] + 0], spins[3*indices[idx] + 1], spins[3*indices[idx] + 2]};

    spins[3*indices[idx] + 0] = Rxx * s[0] + Rxy * s[1] + Rxz * s[2];
    spins[3*indices[idx] + 1] = Ryx * s[0] + Ryy * s[1] + Ryz * s[2];
    spins[3*indices[idx] + 2] = Rzx * s[0] + Rzy * s[1] + Rzz * s[2];
  }

}

void jams::cuda_rotate_spins(jams::MultiArray<double, 2> &spins,
                       const Mat3 &rotation_matrix,
                       const jams::MultiArray<int, 1> &indices) {

  dim3 block_size;
  block_size.x = 128;

  dim3 grid_size;
  grid_size.x = (indices.size() + block_size.x - 1) / block_size.x;

  cuda_rotate_spins_kernel<<<grid_size, block_size>>>(
      spins.device_data(), indices.device_data(), indices.size(),
      rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2],
      rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2],
      rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]);
}
