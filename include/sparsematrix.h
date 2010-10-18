#ifndef __SPARSEMATRIX_H__
#define __SPARSEMATRIX_H__

#include <vector>

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
        val(nnz_guess),
        row(nnz_guess),
        col(nnz_guess)
    {}

    void insert(size_type i, size_type j, _Tp &value);

    double memorySize();
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

#endif // __SPARSEMATRIX_H__
