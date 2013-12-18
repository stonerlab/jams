#include <thrust/extrema.h>
#include <containers/Sparsematrix.h>
#include "cuda_sparse_types.h"
#include "sparsematrix.h"
#include "sparsematrix4d.h"
#include <cstdio>
#include <cmath>

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

void allocate_transfer_csr_4d(jbLib::Sparsematrix<float,4> &Jij, devCSR &
    Jij_dev)
{
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.pointers,(Jij.sizex()+1)*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.coords,(4*Jij.nonZeros())*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&Jij_dev.val,(Jij.nonZeros())*sizeof(float)));

  CUDA_CALL(cudaMemcpy(Jij_dev.pointers,Jij.csrData(),(size_t)((Jij.sizex()+1)*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Jij_dev.coords,Jij.indexData(),(size_t)((4*Jij.nonZeros())*(sizeof(int))),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Jij_dev.val,Jij.valueData(),(size_t)((Jij.nonZeros())*(sizeof(float))),cudaMemcpyHostToDevice));
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

__global__ void dipole_brute_kernel
(
 const float alpha,
 const float beta,
 const float *sf_dev,
 const float *mat_dev,
 float *h_dev, 
 const float *r_dev,
 const float *r_max_dev,
 const bool *pbc_dev,
 const int nspins
)
{
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int i,n;

    if(idx < nspins){
        float sum[3];
        float r_i[3];
        float r_ij[3];
        float s_j[3];

          #pragma unroll
          for(i=0; i<3; ++i){
              r_i[i] = r_dev[3*idx+i];
          }
          
          #pragma unroll
          for(i=0; i<3; ++i){
              sum[i] = 0.0;
          }

        for(n=0; n<nspins; ++n){
            if(n!=idx){
        
              float mus = mat_dev[n*4];
              #pragma unroll
              for(i=0; i<3; ++i){
                s_j[i] = sf_dev[3*n+i];
              }
              
              #pragma unroll
              for(i=0; i<3; ++i){
                  r_ij[i] = (r_dev[3*n+i]-r_i[i]);
                  // check for and perform periodic boundary conditions
                  if(pbc_dev[i] == true){
                      if(fabsf(r_ij[i]) > r_max_dev[i]*0.5f){
                          r_ij[i] = r_ij[i] - copysignf(r_max_dev[i],r_ij[i]);
                      }
                  }
              }


              const float sdotr = s_j[0]*r_ij[0] + s_j[1]*r_ij[1] + s_j[2]*r_ij[2];

              const float r2    = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2];
              const float r     = sqrtf(r2);
#pragma unroll
              for(i=0;i<3;++i){
                  sum[i] = sum[i] + mus*alpha*(3.0*sdotr*r_ij[i] - r2*s_j[i])/(r*r*r*r*r);
              }
            }
        }

#pragma unroll
        for(i=0; i<3; ++i){
          h_dev[3*idx+i] = sum[i];
        }

    }


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
 const float * sf_dev,
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
 const float * sf_dev,
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

      const int4 idx = {coords[4*jj+0],coords[4*jj+1],coords[4*jj+2],coords[4*jj+3]};
       
      float sj[3], sk[3], sl[3];
      
      #pragma unroll
      for(int i=0; i<3; ++i){
        sj[i] = sf_dev[3*idx.y+i];
        //sk[i] = tex1Dfetch(tex_x_float,3*kidx+i);
      }

      #pragma unroll
      for(int i=0; i<3; ++i){
        sk[i] = sf_dev[3*idx.z+i];
        //sk[i] = tex1Dfetch(tex_x_float,3*kidx+i);
      }
      #pragma unroll
      for(int i=0; i<3; ++i){
        sl[i] = sf_dev[3*idx.w+i];
        //sl[i] = tex1Dfetch(tex_x_float,3*lidx+i);
      }

      float k_dot_l = sk[0]*sl[0] + sk[1]*sl[1] + sk[2]*sl[2];
      float j_dot_l = sj[0]*sl[0] + sj[1]*sl[1] + sj[2]*sl[2];
      float j_dot_k = sk[0]*sj[0] + sk[1]*sj[1] + sk[2]*sj[2];

      #pragma unroll
      for(int i=0; i<3; ++i){
        sum[i] += A_ijkl * (sj[i]*k_dot_l + sk[i]*j_dot_l + sl[i]*j_dot_k)/3.0;
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

