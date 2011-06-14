#ifndef __SPARSEMATRIX_H__
#define __SPARSEMATRIX_H__

#include <vector>
#include <cassert>
#include <map>
#include <stdint.h>
#include <cmath>
#include "array.h"
#include "globals.h"
#include <algorithm>
#include <functional>

#define RESTRICT __restrict__

typedef enum { 
  SPARSE_MATRIX_FORMAT_MAP = 0,
  SPARSE_MATRIX_FORMAT_COO = 1,
  SPARSE_MATRIX_FORMAT_CSR = 2
} SparseMatrixFormat_t;

typedef enum { 
  SPARSE_MATRIX_TYPE_GENERAL   = 0,
  SPARSE_MATRIX_TYPE_SYMMETRIC = 1
} SparseMatrixType_t;

typedef enum { 
  SPARSE_MATRIX_MODE_UPPER = 0,
  SPARSE_MATRIX_MODE_LOWER = 1
} SparseMatrixMode_t;


// SparseMatrix stored in CSR format
template <typename _Tp>
class SparseMatrix {

  public:
    typedef int size_type;

    SparseMatrix()
      : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
        matrixType(SPARSE_MATRIX_TYPE_GENERAL),
        matrixMode(SPARSE_MATRIX_MODE_UPPER),
        nrows(0),
        ncols(0),
        nnz_unmerged(0),
        nnz(0),
        val(0),
        row(0),
        col(0) 
    {}


    SparseMatrix(size_type m, size_type n)
      : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
        matrixType(SPARSE_MATRIX_TYPE_GENERAL),
        matrixMode(SPARSE_MATRIX_MODE_UPPER),
        nrows(m),
        ncols(n),
        nnz_unmerged(0),
        nnz(0),
        val(0),
        row(0),
        col(0)
    {}

    // resize clears all data in matrix and prepares for insertion
    // again
    void resize(const size_type m, const size_type n) {
      matrixFormat = SPARSE_MATRIX_FORMAT_MAP;
      nrows = m;
      ncols = n;
      nnz = 0;
      nnz_unmerged = 0;
      val.clear();
      row.clear();
      col.clear();
      non_zeros_unmerged.clear();
    }

    inline SparseMatrixFormat_t getMatrixFormat(){ return matrixFormat; }
    inline void setMatrixType(SparseMatrixType_t type){ matrixType = type; }
    inline SparseMatrixType_t getMatrixType(){ return matrixType; }
    inline void setMatrixMode(SparseMatrixMode_t mode){ matrixMode = mode; }
    inline SparseMatrixMode_t getMatrixMode(){ return matrixMode; }

    void insertValue(size_type i, size_type j, _Tp value);

    void convertMAP2CSR();
    void convertMAP2COO();
    
    inline _Tp*       valPtr() { return &(val[0]); }
    inline size_type* rowPtr() { return &(row[0]); }
    inline size_type* colPtr() { return &(col[0]); }

    inline size_type  nonZero() { return nnz; }
    inline size_type  rows()    { return nrows; }
    inline size_type  cols()    { return ncols; }
    
    // NIST style CSR storage pointers
    inline size_type* RESTRICT ptrB() { return &(row[0]); }
    inline size_type* RESTRICT ptrE() { return &(row[1]); }

    double calculateMemory();
    //void printCSR();
  
  private:

    typedef std::multimap <int64_t,_Tp,std::less<int64_t> > coo_mmp;

    SparseMatrixFormat_t  matrixFormat;
    SparseMatrixType_t    matrixType;
    SparseMatrixMode_t    matrixMode;
    size_type             nrows;
    size_type             ncols;
    size_type             nnz_unmerged;
    size_type             nnz;

    coo_mmp               non_zeros_unmerged;

    std::vector<_Tp>       val;
    std::vector<size_type> row;
    std::vector<size_type> col;

};

// template <typename _Tp>
// void SparseMatrix<_Tp>::printCSR() {
// 
//   if(format==CSR) {
//     for(int i=0; i<nrows+1; ++i) {
//       output.write("%i\n",row[i]);
//     }
//     output.write("\n\n");
//     for(int i=0; i<nnz; ++i) {
//       output.write("%i\n",col[i]);
//     }
//     output.write("\n\n");
//     for(int i=0; i<nnz; ++i) {
//       output.write("%e\n",val[i]);
//     }
//   }
// }

template <typename _Tp>
double SparseMatrix<_Tp>::calculateMemory() {
  const double mb = (1024.0*1024.0);
  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP){
    return ((non_zeros_unmerged.size())*(sizeof(int64_t)+sizeof(_Tp)))/mb;
  } else {
    return (((row.size()+col.size())*sizeof(size_type))+(val.size()*sizeof(_Tp)))/mb;
  }
}

template <typename _Tp>
void SparseMatrix<_Tp>::insertValue(size_type i, size_type j, _Tp value) {
  
  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP) {

    if( ((i < nrows) && (i >= 0)) && ((j < ncols) && (j >= 0)) ) {

      if(matrixType == SPARSE_MATRIX_TYPE_SYMMETRIC) {
        if(matrixMode == SPARSE_MATRIX_MODE_UPPER) {
          if( i > j ) {
            jams_error("Attempted to insert lower matrix element in symmetric upper sparse matrix");
          }
        } else {
          if( j < i ) {
            jams_error("Attempted to insert upper matrix element in symmetric lower sparse matrix");
          }
        }
      }
    } else {
      jams_error("Attempted to insert matrix element outside of matrix size");
    }

    const int64_t index = (static_cast<int64_t>(i)*static_cast<int64_t>(ncols)) + static_cast<int64_t>(j);
    non_zeros_unmerged.insert(typename coo_mmp::value_type(index,value));
    nnz_unmerged++;
  } else {
    jams_error("Can only insert into MAP format sparse matrix");
  }

}


template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2CSR()
{
  typename coo_mmp::iterator nz;

  int64_t index_last,ival,jval,index,row_last;
  
  nnz = 0;
  row.resize(nrows+1);
  col.resize(nnz_unmerged);
  val.resize(nnz_unmerged);
  
  if(nnz_unmerged > 0){

    index_last = -1; // ensure first index is different;

    row_last = 0;
    row[0] = 0;
    

//     std::cerr<<"UNMERGED "<<nnz_unmerged<<std::endl;

    for(nz = non_zeros_unmerged.begin(); nz != non_zeros_unmerged.end(); ++nz){
      index = nz->first;

      if(index < 0){
        jams_error("Negative sparse array index");
      }

      ival = index/static_cast<int64_t>(ncols);

      jval = index - ((ival)*ncols);
      
//       std::cerr<<"i:"<<ival<<" j:"<<jval<<std::endl;

      if(index != index_last){
        index_last = index; // update last index
        nnz++;

        if(ival != row_last){
          // incase there are rows of zeros
          for(int i=row_last; i<ival+1; ++i){
            row[i+1] = nnz-1;
          }
        }
        row_last = ival;
        col[nnz-1] = jval;
        val[nnz-1] = nz->second;

      }else{
        // index is the same as the last, add values
        val[nnz-1] += nz->second;
      }
      non_zeros_unmerged.erase(nz);
    }
  }else{
    for(int i=0;i<nrows+1;++i){
      row[i] = 0;
    }
  }

  col.resize(nnz);
  val.resize(nnz);

  row[nrows] = nnz;
  
  matrixFormat = SPARSE_MATRIX_FORMAT_CSR;
}

template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2COO()
{
  typename coo_mmp::const_iterator nz;

  int64_t index_last,ival,jval,index;
  
  nnz = 0;
  row.resize(nrows+1);
  col.resize(nnz_unmerged);
  val.resize(nnz_unmerged);
  
  if(nnz_unmerged > 0){

    index_last = -1; // ensure first index is different;

    row[0] = 0;
    
    for(nz = non_zeros_unmerged.begin(); nz != non_zeros_unmerged.end(); ++nz){
      index = nz->first;

      if(index < 0){
        jams_error("Negative sparse array index");
      }

      ival = index/static_cast<int64_t>(ncols);

      jval = index - ((ival)*ncols);
      
      if(index != index_last){
        index_last = index; // update last index
        nnz++;

        row[nnz-1] = ival;
        col[nnz-1] = jval;
        val[nnz-1] = nz->second;

      }else{
        // index is the same as the last, add values
        val[nnz-1] += nz->second;
      }
      non_zeros_unmerged.erase(nz);
    }
  }
  
  row.resize(nnz);
  col.resize(nnz);
  val.resize(nnz);

  matrixFormat = SPARSE_MATRIX_FORMAT_COO;
}


void jams_dcsrmv(const char trans[1], const int m, const int k, 
    const double alpha, const char descra[6], const double *val, 
    const int *indx, const int *ptrb, const int *ptre, double *x, 
    const double beta, double * y);

// TEMPORARY HACK FOR CUDA COMPAT
void jams_dcsrmv(const char trans[1], const int m, const int k, 
    const double alpha, const char descra[6], const float *val, 
    const int *indx, const int *ptrb, const int *ptre, double *x, 
    const double beta, double * y);

#endif // __SPARSEMATRIX_H__
