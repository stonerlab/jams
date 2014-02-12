// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SPARSEMATRIX_H
#define JAMS_CORE_SPARSEMATRIX_H

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "core/error.h"

#define RESTRICT __restrict__

typedef enum {
  SPARSE_MATRIX_FORMAT_MAP = 0,
  SPARSE_MATRIX_FORMAT_COO = 1,
  SPARSE_MATRIX_FORMAT_CSR = 2,
  SPARSE_MATRIX_FORMAT_DIA = 3,
} SparseMatrixFormat_t;

typedef enum {
  SPARSE_MATRIX_TYPE_GENERAL   = 0,
  SPARSE_MATRIX_TYPE_SYMMETRIC = 1
} SparseMatrixType_t;

typedef enum {
  SPARSE_MATRIX_MODE_UPPER = 0,
  SPARSE_MATRIX_MODE_LOWER = 1
} SparseMatrixMode_t;

// taken from CUSP
template< class _Tp1, class _Tp2>
bool kv_pair_less(const std::pair<_Tp1, _Tp2>&x, const std::pair<_Tp1, _Tp2>&y){
  return x.first < y.first;
}


///
/// @class SparseMatrix
/// @brief Storage class for sparse matrices.
///
///
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
        val_(0),
        row_(0),
        col_(0),
        dia_offsets(0),
        num_diagonals(0)
    {}


    SparseMatrix(size_type m, size_type n)
      : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
        matrixType(SPARSE_MATRIX_TYPE_GENERAL),
        matrixMode(SPARSE_MATRIX_MODE_UPPER),
        nrows(m),
        ncols(n),
        nnz_unmerged(0),
        nnz(0),
        val_(0),
        row_(0),
        col_(0),
        dia_offsets(0),
        num_diagonals(0)
    {}

    // resize clears all data in matrix and prepares for insertion
    // again
    void resize(const size_type m, const size_type n) {
      matrixFormat = SPARSE_MATRIX_FORMAT_MAP;
      nrows = m;
      ncols = n;
      nnz = 0;
      nnz_unmerged = 0;
      val_.clear();
      row_.clear();
      col_.clear();
      matrixMap.clear();
    }

    inline SparseMatrixFormat_t getMatrixFormat(){ return matrixFormat; }
    inline void setMatrixType(SparseMatrixType_t type){ matrixType = type; }
    inline SparseMatrixType_t getMatrixType(){ return matrixType; }
    inline void setMatrixMode(SparseMatrixMode_t mode){ matrixMode = mode; }
    inline SparseMatrixMode_t getMatrixMode(){ return matrixMode; }

    void insertValue(size_type i, size_type j, _Tp value);

    void convertMAP2CSR();
    void convertMAP2COO();
    void convertMAP2DIA();

    void convertSymmetric2General();

    inline const _Tp val(const size_type &i) const { return val_[i]; }
    inline const size_type row(const size_type &i) const { return row_[i]; }
    inline const size_type col(const size_type &i) const { return col_[i]; }


    inline _Tp*       valPtr() { return &(val_[0]); } ///< @brief Pointer to values array
    inline size_type* rowPtr() { return &(row_[0]); } ///< @brief Pointer to rows array
    inline size_type* colPtr() { return &(col_[0]); } ///< @breif Pointer to columns array

    inline size_type* dia_offPtr() { return &(dia_offsets[0]); }

    inline size_type  nonZero() { return nnz; }      ///< @brief Number of non zero entries in matrix
    inline size_type  rows()    { return nrows; }    ///< @brief Number of rows in matrix
    inline size_type  cols()    { return ncols; }    ///< @brief Number of columns in matrix
    inline size_type  diags()    { return num_diagonals; }


    // NIST style CSR storage pointers
    inline const size_type* RESTRICT ptrB() const { return &(row_[0]); }
    inline const size_type* RESTRICT ptrE() const { return &(row_[1]); }

    double calculateMemory();
    //void printCSR();

  private:

//    typedef std::multimap <int64_t, _Tp, std::less<int64_t> > coo_mmp;
    typedef std::vector< std::pair<int64_t, _Tp> > coo_mmp;

    SparseMatrixFormat_t  matrixFormat;
    SparseMatrixType_t    matrixType;
    SparseMatrixMode_t    matrixMode;
    size_type             nrows;
    size_type             ncols;
    size_type             nnz_unmerged;
    size_type             nnz;

//    coo_mmp               matrixMap;

    coo_mmp matrixMap;

    std::vector<_Tp>       val_;
    std::vector<size_type> row_;
    std::vector<size_type> col_;
    std::vector<size_type> dia_offsets;
    //Array2D<_Tp>           dia_values;
    size_type              num_diagonals;

};

// template <typename _Tp>
// void SparseMatrix<_Tp>::printCSR() {
//
//   if(format==CSR) {
//     for(int i = 0; i<nrows+1; ++i) {
//       output.write("%i\n", row[i]);
//     }
//     output.write("\n\n");
//     for(int i = 0; i<nnz; ++i) {
//       output.write("%i\n", col[i]);
//     }
//     output.write("\n\n");
//     for(int i = 0; i<nnz; ++i) {
//       output.write("%e\n", val[i]);
//     }
//   }
// }

template <typename _Tp>
double SparseMatrix<_Tp>::calculateMemory() {
  const double mb = (1024.0*1024.0);
  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP){
    return ((matrixMap.size())*(sizeof(int64_t)+sizeof(_Tp)))/mb;
  }else if(matrixFormat == SPARSE_MATRIX_FORMAT_DIA){
    return (dia_offsets.size()*sizeof(size_type)+(nrows*num_diagonals)*sizeof(_Tp))/mb;
  } else {
    return (((row_.size()+col_.size())*sizeof(size_type))+(val_.size()*sizeof(_Tp)))/mb;
  }
}

template <typename _Tp>
void SparseMatrix<_Tp>::insertValue(size_type i, size_type j, _Tp value) {

  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP) { // can only insert elements into map formatted matrix

    if( ((i < nrows) && (i >= 0)) && ((j < ncols) && (j >= 0)) ) { // element must be inside matrix boundaries

      if(matrixType == SPARSE_MATRIX_TYPE_SYMMETRIC) {
        if(matrixMode == SPARSE_MATRIX_MODE_UPPER) {
          if( i > j ) {
            jams_error("Attempted to insert lower matrix element in symmetric upper sparse matrix");
          }
        } else {
          if( i < j ) {
            jams_error("Attempted to insert upper matrix element in symmetric lower sparse matrix");
          }
        }
      }
    } else {
      jams_error("Attempted to insert matrix element outside of matrix size");
    }

    // static casts to force 64bit arithmetic
//    const int64_t index = (static_cast<int64_t>(i)*static_cast<int64_t>(ncols)) + static_cast<int64_t>(j);
    const int64_t index = (static_cast<int64_t>(i)*static_cast<int64_t>(ncols)) + static_cast<int64_t>(j);
//    matrixMap.insert(typename coo_mmp::value_type(index, value));
    matrixMap.push_back( std::pair<int64_t, _Tp>(index, value) );

    nnz_unmerged++;
  } else {
    jams_error("Can only insert into MAP format sparse matrix");
  }

}

template <typename _Tp>
void SparseMatrix<_Tp>::convertSymmetric2General() {


  int64_t ival, jval, index, index_new;
  _Tp value;

  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP){
    //if(matrixType == SPARSE_MATRIX_TYPE_SYMMETRIC){
        matrixType = SPARSE_MATRIX_TYPE_GENERAL;
        const int nitems = matrixMap.size();
        matrixMap.reserve(2*nitems);
        for(int i = 0; i<nitems; ++i){
            index = matrixMap[i].first;
            value = matrixMap[i].second;

            // opposite relationship
            jval = index/static_cast<int64_t>(ncols);
            ival = index - ((jval)*ncols);

            if(ival != jval){
                index_new = (static_cast<int64_t>(ival)*static_cast<int64_t>(ncols)) + static_cast<int64_t>(jval);
                matrixMap.push_back(std::pair<int64_t, _Tp>(index_new, value));
                nnz_unmerged++;
            }
        }
    //} else {
        //jams_error("Attempted to generalise a matrix which is already general");
    //}
  }else{
      jams_error("Only a MAP matrix can be generalised");
  }

}


template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2CSR()
{
  typename coo_mmp::const_iterator nz;

  int64_t current_row, current_col, index;
  int64_t previous_index = -1, previous_row = 0;

  nnz = 0;
  row_.resize(nrows+1);
  col_.resize(nnz_unmerged);
  val_.resize(nnz_unmerged);

  std::sort(matrixMap.begin(), matrixMap.end(), kv_pair_less<int64_t, _Tp>);

  if (nnz_unmerged > 0) {
    previous_index = -1;  // ensure first index is different;
    previous_row = 0;
    row_[0] = 0;
    for (nz = matrixMap.begin(); nz != matrixMap.end(); ++nz) {  // iterate matrix map elements
      index = nz->first;  // first part contains row major index
      if (index < 0) {
        jams_error("Negative sparse array index");
      }
      current_row = index/static_cast<int64_t>(ncols);
      current_col = index - ((current_row)*ncols);
      if (index == previous_index) {
        // index is the same as the last, add values
        val_[nnz-1] += nz->second;
      } else {
        if (current_row != previous_row) {
          // fill in row array including any missing entries where there were
          // no row,col values
          for (int i = previous_row+1; i < current_row+1; ++i) {
            row_[i] = nnz;
          }
        }
        col_[nnz] = current_col;
        val_[nnz] = nz->second;
        nnz++;

      }
      previous_row = current_row;
      previous_index = index;  // update last index
    }
  }

  matrixMap.clear();

  col_.resize(nnz);
  val_.resize(nnz);

  // complete the rest of the (empty) row array
  for (int i = previous_row+1; i < nrows+1; ++i) {
    row_[i] = nnz;
  }

  matrixFormat = SPARSE_MATRIX_FORMAT_CSR;
}

template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2COO()
{

  int64_t index_last, ival, jval, index;

  nnz = 0;
  row_.resize(nrows+1);
  col_.resize(nnz_unmerged);
  val_.resize(nnz_unmerged);

  if(nnz_unmerged > 0){

    index_last = -1; // ensure first index is different;

    row_[0] = 0;

    typename coo_mmp::const_iterator elem;
    for(elem = matrixMap.begin(); elem != matrixMap.end(); ++elem){
      index = elem->first;

      if(index < 0){
        jams_error("Negative sparse array index");
      }

      ival = index/static_cast<int64_t>(ncols);

      jval = index - ((ival)*ncols);

      if(index != index_last){
        index_last = index; // update last index
        nnz++;

        row_[nnz-1] = ival;
        col_[nnz-1] = jval;
        val_[nnz-1] = elem->second;

      }else{
        // index is the same as the last, add values
        val_[nnz-1] += elem->second;
      }
    }
  }
  matrixMap.clear();
  row_.resize(nnz);
  col_.resize(nnz);
  val_.resize(nnz);

  matrixFormat = SPARSE_MATRIX_FORMAT_COO;
}

template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2DIA()
{

  int64_t ival, jval, index;

  nnz = 0;

  if(nnz_unmerged > 0){

//    std::sort(matrixMap.begin(), matrixMap.end(), kv_pair_less<int64_t, _Tp>);

    num_diagonals = 0;
    std::vector<int> diag_map(nrows+ncols, 0);

    typename coo_mmp::const_iterator elem;
    for(elem = matrixMap.begin(); elem != matrixMap.end(); ++elem){
      index = elem->first;

      if(index < 0){
        jams_error("Negative sparse array index");
      }

      ival = index/static_cast<int64_t>(ncols);

      jval = index - ((ival)*ncols);

      int map_index = (nrows-ival) + jval;
      if(diag_map[map_index]==0){
        diag_map[map_index] = 1;
        num_diagonals++;
      }
    }

    dia_offsets.resize(num_diagonals);
    val_.resize((nrows*num_diagonals), 0.0);

    for(int i = 0, diag=0; i<(nrows+ncols); ++i){
      if(diag_map[i] == 1){
        diag_map[i] = diag;
        dia_offsets[diag] = i - nrows;
        diag++;
      }
    }

//    for(int i = 0; i<num_diagonals; ++i){
//      std::cout<<dia_offsets[i]<<std::endl;
//    }
//    std::cout<<"\n\n";
//    for(int i = 0; i<nrows; ++i){
//      for(int j = 0; j<num_diagonals; ++j){
//        dia_values[nrows*j+i] = 0.0;
//      }
//    }

    for(elem = matrixMap.begin(); elem != matrixMap.end(); ++elem){
      index = elem->first;

      ival = index/static_cast<int64_t>(ncols);
      jval = index - ((ival)*ncols);
      int map_index = (nrows - ival) + jval;
      int diag = diag_map[map_index];
      val_[nrows*diag+ival] += elem->second;

//      std::cout<<ival<<" , "<<jval<<" , "<<diag<<" , "<<val[nrows*diag+ival]<<std::endl;
      nnz++;
    }
  }

  matrixMap.clear();

  matrixFormat = SPARSE_MATRIX_FORMAT_DIA;
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

#endif // JAMS_CORE_SPARSEMATRIX_H
