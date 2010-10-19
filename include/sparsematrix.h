#ifndef __SPARSEMATRIX_H__
#define __SPARSEMATRIX_H__

#include <vector>
#include <cassert>
#include "globals.h"

#define RESTRICT __restrict__

enum SparseMatrixFormat{ COO, CSR };

// SparseMatrix stored in CSR format
template <typename _Tp>
class SparseMatrix {

  public:
    typedef int size_type;

    SparseMatrix()
      : format(COO),
        nrows(0),
        ncols(0),
        nnz(0),
        val(0),
        row(0),
        col(0) 
    {}


    SparseMatrix(size_type m, size_type n, size_type nnz_guess)
      : format(COO),
        nrows(m),
        ncols(n),
        nnz(0),
        val(0),
        row(0),
        col(0)
    {val.reserve(nnz_guess); row.reserve(nnz_guess); col.reserve(nnz_guess);}

    void resize(size_type m, size_type n, size_type nnz_guess) {
      val.empty();
      col.empty();
      row.empty();
      format = COO;
      nrows = m;
      ncols = n;
      nnz = 0;
      val.reserve(nnz_guess);
      col.reserve(nnz_guess);
      row.reserve(nnz_guess);
    }

    void insert(size_type i, size_type j, _Tp value);

    double memorySize();

    void printCSR();

    void coocsr();

    inline int nonzero() { return nnz; }
    
    inline _Tp* ptrVal() {
      return &(val[0]);
    }

    inline size_type* ptrCol() {
      return &(col[0]);
    }

    inline size_type* ptrRow() {
      return &(row[0]);
    }


    // NIST style CSR storage pointers
    inline size_type* RESTRICT ptrB() {
      return &(row[0]);
    }

    inline size_type* RESTRICT ptrE() {
      return &(row[1]);
    }
  
  private:
    SparseMatrixFormat     format;
    size_type              nrows;
    size_type              ncols;
    size_type              nnz;
    std::vector<_Tp>    val;
    std::vector<size_type> row;
    std::vector<size_type> col;

};

template <typename _Tp>
void SparseMatrix<_Tp>::printCSR() {

  if(format==CSR) {
    for(int i=0; i<nrows+1; ++i) {
      output.write("%i\n",row[i]);
    }
    output.write("\n\n");
    for(int i=0; i<nnz; ++i) {
      output.write("%i\n",col[i]);
    }
  }
}

template <typename _Tp>
double SparseMatrix<_Tp>::memorySize() {
  const double mb = (1024.0*1024.0);
  return (((row.size()+col.size())*sizeof(size_type))+(val.size()*sizeof(_Tp)))/mb;
}

template <typename _Tp>
void SparseMatrix<_Tp>::insert(size_type i, size_type j, _Tp value) {
  if(format != COO) {
    jams_error("Can only insert into COO format sparse matrix");
  }

  assert(i < nrows);
  assert(j < ncols);

  row.push_back(i);
  col.push_back(j);
  val.push_back(value);

  nnz++;
}

//coo-csr conversion from fortran sparsekit
// http://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit/sparsekit.f
template <typename _Tp>
void SparseMatrix<_Tp>::coocsr()
{
  if(format == CSR) {
    output.write("WARNING: Cannot convert SparseMatrix");
    output.write("Matrix is alread in CSR format");
    return;
  }

  std::vector<size_type>  csrrow((nrows+1),0);
  std::vector<size_type>  csrcol(nnz,0);
  std::vector<_Tp>     csrval(nnz);

  // determine row lengths
  for(int k=0; k<nnz; ++k) {
    csrrow[row[k]]++;
  }


  // starting poition of each row
  int p=0;
  for(int j=0; j<(nrows+1); ++j) {
    int p0 = csrrow[j];
    csrrow[j] = p;
    p = p+p0;
  }

  // go through the structure once more, fill in the output matrix

  for(int k=0; k<nnz; ++k) {
    const int i = row[k];
    const int j = col[k];
    const _Tp x = val[k];
    const _Tp ia = csrrow[i];
    csrval[ia] = x;
    csrcol[ia] = j;
    csrrow[i] = ia+1;
  }

  // shift back csrrow
  for(int j=0; j<nrows+1; ++j) {
    const int idx = nrows+1-j;
    csrrow[idx] = csrrow[idx-1];
  }
  csrrow[0] = 0;

  row.resize(nrows+1);
  row.swap(csrrow);
  val.swap(csrval);
  col.swap(csrcol);

  format = CSR;
}


#endif // __SPARSEMATRIX_H__
