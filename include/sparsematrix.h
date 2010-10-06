#ifndef __SPARSEMATRIX_H__
#define __SPARSEMATRIX_H__

#include "globals.h"

#include <vector>

enum SparseMatrixFormat{ COO, CSR };

// SparseMatrix stored in CSR format
class SparseMatrix {

  public:
    typedef unsigned int size_type;

    SparseMatrix(size_type m, size_type n, size_type nnz_guess) {
      dim[0] = m; dim[1] = n;
      val.reserve(nnz_guess);
      row.reserve(nnz_guess);
      col.reserve(nnz_guess);
      nnz=0;
      format=COO;
    }


    void insert(size_type i, size_type j, double &value);

    double memorySize();
    void coocsr();


    // NIST style CSR storage pointers
    inline double& RESTRICT ptrB() {
      return &(row[0]);
    }

    inline double& RESTRICT ptrE() {
      return &(row[1]);
    }

  private:
    SparseMatrixFormat     format;

    size_type              dim[2]; // number of rows, cols
    size_type              nnz;
    std::vector<double>    val;
    std::vector<size_type> row;
    std::vector<size_type> col;
};

#endif // __SPARSEMATRIX_H__
