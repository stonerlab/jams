#include "sparsematrix.h"

#include "globals.h"

#include <valarray>
#include <cassert>

double SparseMatrix::memorySize() {
  const double mb = (1024.0*1024.0);
  return (((row.size()+col.size())*sizeof(size_type))+(val.size()*sizeof(double)))/mb;
}

void SparseMatrix::insert(size_type i, size_type j, double &value) {
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
void SparseMatrix::coocsr()
{
  if(format == CSR) {
    output.write("WARNING: Cannot convert SparseMatrix");
    output.write("Matrix is alread in CSR format");
    return;
  }

  std::vector<size_type>  csrrow((nrows+1),0);
  std::vector<size_type>  csrcol(nnz,0);
  std::vector<double>     csrval(nnz,0.0);

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
    const double x = val[k];
    const double ia = csrrow[i];
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
  row = csrrow;
  val = csrval;
  col = csrcol;

  format = CSR;
}
