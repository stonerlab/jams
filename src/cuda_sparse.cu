#include <thrust/extrema.h>
#include "cuda_sparse_types.h"
#include "sparsematrix.h"
#include "sparsematrix4d.h"
#include <cstdio>

#define DIA_BLOCK_SIZE 256
#define CSR_4D_BLOCK_SIZE 64

/*texture<float,1> tex_x_float;*/

void allocate_transfer_dia(SparseMatrix<float> &Jij, devDIA &Jij_dev)
{
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.row,(Jij.diags())*sizeof(int)));
  CUDA_CALL(cudaMallocPitch((void**)&Jij_dev.val,&Jij_dev.pitch,(Jij.rows())*sizeof(float),Jij.diags()));
  
  CUDA_CALL(cudaMemcpy(Jij_dev.row,Jij.dia_offPtr(),(size_t)((Jij.diags())*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(Jij_dev.val,Jij_dev.pitch,Jij.valPtr(),Jij.rows()*sizeof(float),Jij.rows()*sizeof(float),Jij.diags(),cudaMemcpyHostToDevice));
  Jij_dev.pitch = Jij_dev.pitch/sizeof(float);
}

void free_dia(devDIA &Jij_dev)
{
  CUDA_CALL(cudaFree(Jij_dev.row));
  CUDA_CALL(cudaFree(Jij_dev.col));
  CUDA_CALL(cudaFree(Jij_dev.val));
}

void allocate_transfer_csr_4d(SparseMatrix4D<float> &Jij, devCSR &
    Jij_dev)
{
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.pointers,(Jij.size(0)+1)*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.coords,(3*Jij.nonZero())*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.val,(Jij.nonZero())*sizeof(float)));

  CUDA_CALL(cudaMemcpy(Jij_dev.pointers,Jij.pointersPtr(),(size_t)((Jij.size(0)+1)*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Jij_dev.coords,Jij.cooPtr(),(size_t)((3*Jij.nonZero())*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Jij_dev.val,Jij.valPtr(),(size_t)((Jij.nonZero())*(sizeof(float))),cudaMemcpyHostToDevice));
}

void free_csr_4d(devCSR &Jij_dev)
{
  CUDA_CALL(cudaFree(Jij_dev.pointers));
  CUDA_CALL(cudaFree(Jij_dev.coords));
  CUDA_CALL(cudaFree(Jij_dev.val));
}


__global__ void biquadratic_scalar_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 float * sf_dev,
 float * h_dev)
{
  __shared__ int offsets[DIA_BLOCK_SIZE];

  const int thread_id = DIA_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = DIA_BLOCK_SIZE * gridDim.x;

  for(int base = 0; base < ndiag; base += DIA_BLOCK_SIZE)
  {
      // read a chunk of the diagonal offsets into shared memory
      const int chunk_size = thrust::min(int(DIA_BLOCK_SIZE), ndiag - base);

      if(threadIdx.x < chunk_size) {
          offsets[threadIdx.x] = dia_offsets[base + threadIdx.x];
      }
  
      __syncthreads();
 
      // process chunk
      for(int row = thread_id; row < nrows; row += grid_size)
      {
        const int rowOff = 3*row;
        float sum[3];
        
        if(base == 0){
          // NOTE: floating point comparison avoids reading h_dev[] for
          // special case
          if(beta == 0.0){
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = 0.0;
            }
          } else {
            // read initial sum values
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = beta*h_dev[rowOff+i];
            }
          }
        } else {
          // outside base 0 use existing values
          #pragma unroll
          for(int i=0; i<3; ++i){
            sum[i] = h_dev[rowOff+i];
          }
        }

        // index into values array
        int idxUp  = row + pitch * base;

        for(int n = 0; n < chunk_size; n++)
        {
            const int colUp  = row + offsets[n];
            const int colLow = row - offsets[n];

            if(colLow >= row && colLow < ncols) {
              const int si = rowOff; // 3*row
              const int sj = 3*colLow;

              const float prod = sf_dev[si+0]*sf_dev[sj+0] 
                              + sf_dev[si+1]*sf_dev[sj+1]
                              + sf_dev[si+2]*sf_dev[sj+2];
                
              const float A_ij = alpha*prod*dia_values[pitch*(base+n)+colLow];

              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }
            if(colUp >= 0 && colUp < row) {
              const int si = rowOff; // 3*row
              const int sj = 3*colUp;

              const float prod = sf_dev[si+0]*sf_dev[sj+0] 
                              + sf_dev[si+1]*sf_dev[sj+1]
                              + sf_dev[si+2]*sf_dev[sj+2];
                
              const float A_ij = alpha*prod*dia_values[idxUp];
              
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }

            idxUp += pitch;
        }

        for(int i=0; i<3; ++i){
          h_dev[rowOff+i] = sum[i];
        }
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }

}

__global__ void bilinear_scalar_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 float * sf_dev,
 float * h_dev)
{
  __shared__ int offsets[DIA_BLOCK_SIZE];

  const int thread_id = DIA_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = DIA_BLOCK_SIZE * gridDim.x;

  for(int base = 0; base < ndiag; base += DIA_BLOCK_SIZE)
  {
      // read a chunk of the diagonal offsets into shared memory
      const int chunk_size = thrust::min(int(DIA_BLOCK_SIZE), ndiag - base);

      if(threadIdx.x < chunk_size) {
          offsets[threadIdx.x] = dia_offsets[base + threadIdx.x];
      }
  
      __syncthreads();
 
      // process chunk
      for(int row = thread_id; row < nrows; row += grid_size)
      {
        const int rowOff = 3*row;
        float sum[3];
        if(base == 0){
          // NOTE: floating point comparison avoids reading h_dev[] for
          // special case
          if(beta == 0.0){
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = 0.0;
            }
          } else {
            // read initial sum values
            #pragma unroll
            for(int i=0; i<3; ++i){
              sum[i] = beta*h_dev[rowOff+i];
            }
          }
        } else {
          // outside base 0 use existing values
          #pragma unroll
          for(int i=0; i<3; ++i){
            sum[i] = h_dev[rowOff+i];
          }
        }

        // index into values array
        int idxUp  = row + pitch * base;

        for(int n = 0; n < chunk_size; n++)
        {
            const int colUp  = row + offsets[n];
            const int colLow = row - offsets[n];

            if(colLow >= row && colLow < ncols) {
              const int sj = 3*colLow;

              const float A_ij = alpha*dia_values[pitch*(base+n)+colLow];

              #pragma unroll
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }
            if(colUp >= 0 && colUp < row) {
              const int sj = 3*colUp;

              const float A_ij = alpha*dia_values[idxUp];
              
              #pragma unroll
              for(int i=0; i<3; ++i){
                sum[i] += A_ij * sf_dev[sj+i];
              }
            }

            idxUp += pitch;
        }

        #pragma unroll
        for(int i=0; i<3; ++i){
          h_dev[rowOff+i] = sum[i];
        }
      }

      // wait until all threads are done reading offsets 
      __syncthreads();
  }

}



__global__ void spmv_dia_kernel
(const int nrows,
 const int ncols,
 const int ndiag,
 const int pitch,
 const float alpha,
 const float beta,
 const int * dia_offsets,
 const float * dia_values,
 const float * x,
 float * y)
{

/*

  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < nrows){
    float dot=0;
    for(int n=0;n<ndiag;++n){
      int colUp  = row+dia_offsets[n];
      int colLow = row-dia_offsets[n];
      float valUp  = dia_values[nrows*n+row];
      float valLow = dia_values[nrows*n+colLow];

      if(colLow >=row && colLow < ncols)
        dot += valLow*x[colLow];

      if(colUp >=0 && colUp < row)
        dot += valUp*x[colUp];
    }
    y[row] = dot;
  }

*/

  __shared__ int offsets[DIA_BLOCK_SIZE];

  const int thread_id = DIA_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = DIA_BLOCK_SIZE * gridDim.x;

  if(ndiag == 0){
      for(int row = thread_id; row < nrows; row += grid_size){
        y[row] = 0.0;
      }
  } else {
    for(int base = 0; base < ndiag; base += DIA_BLOCK_SIZE)
    {
        // read a chunk of the diagonal offsets into shared memory
        const int chunk_size = thrust::min(int(DIA_BLOCK_SIZE), ndiag - base);

        if(threadIdx.x < chunk_size) {
            offsets[threadIdx.x] = dia_offsets[base + threadIdx.x];
        }
    
        __syncthreads();
   
        // process chunk
        for(int row = thread_id; row < nrows; row += grid_size)
        {
            float sum;
            if(base == 0){
              // NOTE: floating point comparison avoids reading h_dev[] for
              // special case
              if(beta == 0.0){
                sum=0.0;
              }else{
                sum = beta*y[row];
              }
            } else {
              sum = y[row];
            }
    
            // index into values array
            int idxUp  = row + pitch * base;
    
            for(int n = 0; n < chunk_size; n++)
            {
                const int colUp  = row + offsets[n];
                const int colLow = row - offsets[n];

                if(colLow >= row && colLow < ncols) {
                  const float A_ij = alpha*dia_values[pitch*(base+n)+colLow];
                  sum += A_ij * x[colLow];
                  //sum += A_ij * tex1Dfetch(tex_x_float,colLow);
                }
                if(colUp >= 0 && colUp < row) {
                  const float A_ij = alpha*dia_values[idxUp];
                  sum += A_ij * x[colUp];
                  //sum += A_ij * tex1Dfetch(tex_x_float,colUp);
                }

                idxUp += pitch;
            }
    
            y[row] = sum;
        }

        // wait until all threads are done reading offsets 
        __syncthreads();
    }
  }
}
__global__ void fourspin_scalar_csr_kernel
(const int num_rows,
 const int nspins,
 const float alpha,
 const float beta,
 const int * pointers,
 const int * coords,
 const float * val,
 const float * sf_dev,
 float * h_dev)
{

  const int thread_id = CSR_4D_BLOCK_SIZE * blockIdx.x + threadIdx.x;
  const int grid_size = CSR_4D_BLOCK_SIZE * gridDim.x;

  for(int row = thread_id; row < num_rows; row += grid_size)
  {
    const int row_start = pointers[row];
    const int row_end   = pointers[row+1];

    float sum[3] = {0.0, 0.0, 0.0};

    for(int jj = row_start; jj < row_end; ++jj){
              
      const float A_ijkl = alpha*val[jj];

      const int jidx = coords[3*jj+0];
      const int kidx = coords[3*jj+1];
      const int lidx = coords[3*jj+2];
       
      float sk[3], sl[3];
      
      #pragma unroll
      for(int i=0; i<3; ++i){
        sk[i] = sf_dev[3*kidx+i];
        //sk[i] = tex1Dfetch(tex_x_float,3*kidx+i);
      }
      #pragma unroll
      for(int i=0; i<3; ++i){
        sl[i] = sf_dev[3*lidx+i];
        //sl[i] = tex1Dfetch(tex_x_float,3*lidx+i);
      }

      float k_dot_l = sk[0]*sl[0] + sk[1]*sl[1] + sk[2]*sl[2];

      #pragma unroll
      for(int i=0; i<3; ++i){
        sum[i] += A_ijkl * sf_dev[3*jidx+i]*k_dot_l;
        //sum[i] += A_ijkl * tex1Dfetch(tex_x_float,3*jidx+i)*k_dot_l;
      }
    }

    if(beta == 0.0){ // NOTE: floating point comparison
      #pragma unroll
      for(int i=0; i<3; ++i){
        h_dev[3*row+i] = sum[i];
      }
    }else{
      #pragma unroll
      for(int i=0; i<3; ++i){
        h_dev[3*row+i] = beta*h_dev[3*row+i] + sum[i];
      }
    }
  }
}
