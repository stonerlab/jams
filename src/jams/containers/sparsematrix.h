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

#include "jams/helpers/error.h"

#define RESTRICT __restrict__

typedef enum
{
  SPARSE_MATRIX_FORMAT_MAP = 0,
  SPARSE_MATRIX_FORMAT_COO = 1,
  SPARSE_MATRIX_FORMAT_CSR = 2,
  SPARSE_MATRIX_FORMAT_DIA = 3,
} sparse_matrix_format_t;

// these definition mirror those in mkl_spblas.h
#ifndef HAS_MKL

typedef enum
{
  SPARSE_MATRIX_TYPE_GENERAL            = 20,
  SPARSE_MATRIX_TYPE_SYMMETRIC          = 21,
  SPARSE_MATRIX_TYPE_HERMITIAN          = 22,
  SPARSE_MATRIX_TYPE_TRIANGULAR         = 23,
  SPARSE_MATRIX_TYPE_DIAGONAL           = 24,
  SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25,
  SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26
} sparse_matrix_type_t;

typedef enum
{
  SPARSE_FILL_MODE_LOWER = 40,
  SPARSE_FILL_MODE_UPPER = 41
} sparse_fill_mode_t;

#else

#include <mkl_spblas.h>

#endif

// taken from CUSP
template< class _Tp1, class _Tp2>
bool kv_pair_less(const std::pair<_Tp1, _Tp2>&x, const std::pair<_Tp1, _Tp2>&y){
  return x.first < y.first;
}

template< class _Tp1, class _Tp2>
bool kv_pair_greater(const std::pair<_Tp1, _Tp2>&x, const std::pair<_Tp1, _Tp2>&y){
  return x.first > y.first;
}


///
/// @class SparseMatrix
/// @brief Storage class for sparse matrices.
///
///
template <typename _Tp>
class SparseMatrix {

  public:
    typedef int      size_type;
    typedef uint64_t hash_type;

    SparseMatrix()
      : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
        matrixType(SPARSE_MATRIX_TYPE_GENERAL),
        matrixMode(SPARSE_FILL_MODE_UPPER),
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
        matrixMode(SPARSE_FILL_MODE_UPPER),
        nrows(m),
        ncols(n),
        nnz_unmerged(0),
        nnz(0),
        val_(0),
        row_(0),
        col_(0),
        dia_offsets(0),
        num_diagonals(0)
    {
      if(hash_type(m)*hash_type(n) > std::numeric_limits<hash_type>::max()) {
        throw std::runtime_error("sparsematrix.h:  maximum hash size is not large enough for the requested matrix size");
      }
    }

    // resize clears all data in matrix and prepares for insertion
    // again
    void resize(const size_type m, const size_type n) {
      if(hash_type(m)*hash_type(n) > std::numeric_limits<hash_type>::max()) {
        throw std::runtime_error("sparsematrix.h:  maximum hash size is not large enough for the requested matrix size");
      }

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

    inline sparse_matrix_format_t getMatrixFormat() const { return matrixFormat; }
    inline void setMatrixType(sparse_matrix_type_t type){ matrixType = type; }
    inline sparse_matrix_type_t getMatrixType() const { return matrixType; }
    inline void setMatrixMode(sparse_fill_mode_t mode){ matrixMode = mode; }
    inline sparse_fill_mode_t getMatrixMode() const { return matrixMode; }

    void insertValue(size_type i, size_type j, _Tp value);

    void convertMAP2CSR();
    void convertMAP2COO();
    void convertMAP2DIA();

    void convertSymmetric2General();

    inline _Tp val(const size_type &i) const { return val_[i]; }
    inline size_type row(const size_type &i) const { return row_[i]; }
    inline size_type col(const size_type &i) const { return col_[i]; }


    inline _Tp*       valPtr() { return &(val_[0]); } ///< @brief Pointer to values array
    inline size_type* rowPtr() { return &(row_[0]); } ///< @brief Pointer to rows array
    inline size_type* colPtr() { return &(col_[0]); } ///< @breif Pointer to columns array
    inline size_type* dia_offPtr() { return &(dia_offsets[0]); }

    inline const _Tp*       valPtr() const { return &(val_[0]); } ///< @brief Pointer to values array
    inline const size_type* rowPtr() const { return &(row_[0]); } ///< @brief Pointer to rows array
    inline const size_type* colPtr() const { return &(col_[0]); } ///< @breif Pointer to columns array
    inline const size_type* dia_offPtr() const { return &(dia_offsets[0]); }

    inline size_type  nonZero() const { return nnz; }      ///< @brief Number of non zero entries in matrix
    inline size_type  rows()    const { return nrows; }    ///< @brief Number of rows in matrix
    inline size_type  cols()    const { return ncols; }    ///< @brief Number of columns in matrix
    inline size_type  diags()    const { return num_diagonals; }


    // NIST style CSR storage pointers
    inline size_type* RESTRICT ptrB() { return &(row_[0]); }
    inline size_type* RESTRICT ptrE() { return &(row_[1]); }

    inline const size_type* RESTRICT ptrB() const { return &(row_[0]); }
    inline const size_type* RESTRICT ptrE() const { return &(row_[1]); }

    void reserveMemory(size_type n);
    double calculateMemory();
    //void printCSR();

  private:

//    typedef std::multimap <hash_type, _Tp, std::less<hash_type> > coo_mmp;
    typedef std::vector< std::pair<hash_type, _Tp> > coo_mmp;

    sparse_matrix_format_t  matrixFormat;
    sparse_matrix_type_t    matrixType;
    sparse_fill_mode_t    matrixMode;
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
    hash_type             max_hash_;
    hash_type             max_index_;

};

// template <typename _Tp>
// void SparseMatrix<_Tp>::printCSR() {
//
//   if(format==CSR) {
//     for(int i = 0; i<nrows+1; ++i) {
//       output->write("%i\n", row[i]);
//     }
//     output->write("\n\n");
//     for(int i = 0; i<nnz; ++i) {
//       output->write("%i\n", col[i]);
//     }
//     output->write("\n\n");
//     for(int i = 0; i<nnz; ++i) {
//       output->write("%e\n", val[i]);
//     }
//   }
// }

template <typename _Tp>
double SparseMatrix<_Tp>::calculateMemory() {
  const double mb = (1024.0*1024.0);
  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP){
    return ((matrixMap.size())*(sizeof(hash_type)+sizeof(_Tp)))/mb;
  }else if(matrixFormat == SPARSE_MATRIX_FORMAT_DIA){
    return (dia_offsets.size()*sizeof(size_type)+(nrows*num_diagonals)*sizeof(_Tp))/mb;
  } else {
    return (((row_.size()+col_.size())*sizeof(size_type))+(val_.size()*sizeof(_Tp)))/mb;
  }
}

template <typename _Tp>
void SparseMatrix<_Tp>::reserveMemory(size_type n) {
  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP) {
    matrixMap.reserve(n);
  }
}

template <typename _Tp>
void SparseMatrix<_Tp>::insertValue(size_type i, size_type j, _Tp value) {

  if (nnz_unmerged == std::numeric_limits<size_type>::max() - 1) {
    throw std::runtime_error("sparsematrix.h: number of non zero elements is too large for the size_type");
  }

  if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP) { // can only insert elements into map formatted matrix

    if( ((i < nrows) && (i >= 0)) && ((j < ncols) && (j >= 0)) ) { // element must be inside matrix boundaries

      if(matrixType == SPARSE_MATRIX_TYPE_SYMMETRIC) {
        if(matrixMode == SPARSE_FILL_MODE_UPPER) {
          if( i > j ) {
            die("Attempted to insert lower matrix element in symmetric upper sparse matrix");
          }
        } else {
          if( i < j ) {
            die("Attempted to insert upper matrix element in symmetric lower sparse matrix");
          }
        }
      }
    } else {
      die("Attempted to insert matrix element (%d, %d) outside of matrix size (%d, %d)", i, j, nrows, ncols);
    }

    // static casts to force 64bit arithmetic
//    const hash_type index = (static_cast<hash_type>(i)*static_cast<hash_type>(ncols)) + static_cast<hash_type>(j);
    const hash_type index = (static_cast<hash_type>(i)*static_cast<hash_type>(ncols)) + static_cast<hash_type>(j);
//    matrixMap.insert(typename coo_mmp::value_type(index, value));
    matrixMap.push_back( std::pair<hash_type, _Tp>(index, value) );

    nnz_unmerged++;
  } else {
    die("Can only insert into MAP format sparse matrix");
  }

}

template <typename _Tp>
void SparseMatrix<_Tp>::convertSymmetric2General() {


  hash_type ival, jval, index, index_new;
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
            jval = index/static_cast<hash_type>(ncols);
            ival = index - ((jval)*ncols);

            if(ival != jval){
                index_new = (static_cast<hash_type>(ival)*static_cast<hash_type>(ncols)) + static_cast<hash_type>(jval);
                matrixMap.push_back(std::pair<hash_type, _Tp>(index_new, value));
                nnz_unmerged++;
            }
        }
  }else{
    die("Only a MAP matrix can be generalised");
  }

}


template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2CSR()
{
  hash_type current_row, current_col, index;
  hash_type previous_index = 0, previous_row = 0;

  nnz = 0;
  row_.resize(nrows+1);

  std::sort(matrixMap.begin(), matrixMap.end(), kv_pair_greater<hash_type, _Tp>);

  if (nnz_unmerged > 0) {
    previous_row = 0;
    row_[0] = 0;

    int pop_counter = 0;
    while (!matrixMap.empty()) {
      auto nz = matrixMap.back();
      matrixMap.pop_back();
      pop_counter++;

      index = nz.first;  // first part contains row major index

      // if (index < 0) {
      //   die("Negative sparse array index");
      // }

      current_row = index/static_cast<hash_type>(ncols);
      current_col = index - ((current_row)*ncols);
      if (nnz != 0 && index == previous_index) { // first value has no previous value
        // index is the same as the last, add values
        val_[nnz-1] += nz.second;
      } else {
        if (current_row != previous_row) {
          // fill in row array including any missing entries where there were
          // no row,col values
          for (int i = previous_row+1; i < current_row+1; ++i) {
            row_[i] = nnz;
          }
        }
        col_.push_back(current_col);
        val_.push_back(nz.second);
        nnz++;
      }
      previous_row = current_row;
      previous_index = index;  // update last index
    }
  }

  // clear matrixMap and reduce memory to zero
  coo_mmp().swap(matrixMap);
 
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

  hash_type index_last, ival, jval, index;

  nnz = 0;
  row_.resize(nrows+1);
  col_.resize(nnz_unmerged);
  val_.resize(nnz_unmerged);

  if(nnz_unmerged > 0){

    row_[0] = 0;

    typename coo_mmp::const_iterator elem;
    for(elem = matrixMap.begin(); elem != matrixMap.end(); ++elem){
      index = elem->first;

      // if(index < 0){
      //   die("Negative sparse array index");
      // }

      ival = index/static_cast<hash_type>(ncols);

      jval = index - ((ival)*ncols);

      if(nnz != 0 && index != index_last){
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
  // clear matrixMap and reduce memory to zero
  coo_mmp().swap(matrixMap);

  row_.resize(nnz);
  col_.resize(nnz);
  val_.resize(nnz);

  matrixFormat = SPARSE_MATRIX_FORMAT_COO;
}

template <typename _Tp>
void SparseMatrix<_Tp>::convertMAP2DIA()
{

  hash_type ival, jval, index;

  nnz = 0;

  if(nnz_unmerged > 0){

//    std::sort(matrixMap.begin(), matrixMap.end(), kv_pair_less<hash_type, _Tp>);

    num_diagonals = 0;
    std::vector<int> diag_map(nrows+ncols, 0);

    typename coo_mmp::const_iterator elem;
    for(elem = matrixMap.begin(); elem != matrixMap.end(); ++elem){
      index = elem->first;

      // if(index < 0){
      //   die("Negative sparse array index");
      // }

      ival = index/static_cast<hash_type>(ncols);

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

      ival = index/static_cast<hash_type>(ncols);
      jval = index - ((ival)*ncols);
      int map_index = (nrows - ival) + jval;
      int diag = diag_map[map_index];
      val_[nrows*diag+ival] += elem->second;

//      std::cout<<ival<<" , "<<jval<<" , "<<diag<<" , "<<val[nrows*diag+ival]<<std::endl;
      nnz++;
    }
  }

  // clear matrixMap and reduce memory to zero
  coo_mmp().swap(matrixMap);

  matrixFormat = SPARSE_MATRIX_FORMAT_DIA;
}


void jams_dcsrmv(const char trans[1], const int m, const int k,
    const double alpha, const char descra[6], const double *val,
    const int *indx, const int *ptrb, const int *ptre, const double *x,
    const double beta, double * y);

// TEMPORARY HACK FOR CUDA COMPAT
void jams_dcsrmv(const char trans[1], const int m, const int k,
    const double alpha, const char descra[6], const float *val,
    const int *indx, const int *ptrb, const int *ptre, const double *x,
    const double beta, double * y);

void jams_scsrmv(const char trans[1], const int m, const int k,
    const double alpha, const char descra[6], const float *val,
    const int *indx, const int *ptrb, const int *ptre, const double *x,
    const double beta, double * y);

#endif // JAMS_CORE_SPARSEMATRIX_H
