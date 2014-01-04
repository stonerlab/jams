#ifndef JAMS_CORE_SPARSEMATRIX4D_H
#define JAMS_CORE_SPARSEMATRIX4D_H

#include <vector>
#include <cassert>
#include <map>
#include <stdint.h>
#include <cmath>
#include "error.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <containers/array.h>

#define RESTRICT __restrict__

///
/// @class SparseMatrix
/// @brief Storage class for sparse matrices.
/// 
///
template <typename _Tp>
class SparseMatrix4D {

    public:
        typedef int size_type;

        SparseMatrix4D()
            : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
            matrixType(SPARSE_MATRIX_TYPE_GENERAL),
            matrixMode(SPARSE_MATRIX_MODE_UPPER),
            dim(4,0),
            nnz_unmerged(0),
            nnz(0),
            i_idx(0),
            j_idx(0),
            k_idx(0),
            l_idx(0),
            pointers(0),
            coords(0,0),
            val(0)
    {}


        SparseMatrix4D(size_type m, size_type n, size_type p, size_type q)
            : matrixFormat(SPARSE_MATRIX_FORMAT_MAP),
            matrixType(SPARSE_MATRIX_TYPE_GENERAL),
            matrixMode(SPARSE_MATRIX_MODE_UPPER),
            dim(4,0),
            nnz_unmerged(0),
            nnz(0),
            i_idx(0),
            j_idx(0),
            k_idx(0),
            l_idx(0),
            pointers(0),
            coords(0,0),
            val(0)
    {dim[0] = m; dim[1] = n; dim[2] = p; dim[3] = q;}

        // resize clears all data in matrix and prepares for insertion
        // again
        void resize(const size_type m, const size_type n, const size_type p, const size_type q) {
            matrixFormat = SPARSE_MATRIX_FORMAT_MAP;
            dim[0] = m,
                dim[1] = n,
                dim[2] = p,
                dim[3] = q,
                nnz = 0;
            nnz_unmerged = 0;
            val.clear();
            i_idx.clear();
            j_idx.clear();
            k_idx.clear();
            l_idx.clear();
        }

        inline SparseMatrixFormat_t getMatrixFormat(){ return matrixFormat; }
        inline void setMatrixType(SparseMatrixType_t type){ matrixType = type; }
        inline SparseMatrixType_t getMatrixType(){ return matrixType; }
        inline void setMatrixMode(SparseMatrixMode_t mode){ matrixMode = mode; }
        inline SparseMatrixMode_t getMatrixMode(){ return matrixMode; }

        void insertValue(size_type i, size_type j, size_type k, size_type l, _Tp value);

        void convertMAP2COO();
        void convertMAP2CSR();

        inline _Tp*       valPtr() { return &(val[0]); } ///< @brief Pointer to values array
        inline size_type*       pointersPtr() { return pointers.data(); } ///< @brief Pointer to values array
        inline size_type* cooPtr() { return coords.data(); } ///< @brief Pointer to rows array


        inline size_type  nonZero() { return nnz; }      ///< @brief Number of non zero entries in matrix
        inline size_type  size(const size_type i) { return dim[i]; }

        double calculateMemory();

    private:

        typedef std::vector< std::pair<int64_t,_Tp> > coo_mmp;

        SparseMatrixFormat_t  matrixFormat;
        SparseMatrixType_t    matrixType;
        SparseMatrixMode_t    matrixMode;
        std::vector<size_type>  dim;
        size_type             nnz_unmerged;
        size_type             nnz;

        coo_mmp matrixMap;

        std::vector<size_type> i_idx;
        std::vector<size_type> j_idx;
        std::vector<size_type> k_idx;
        std::vector<size_type> l_idx;

        jblib::Array<size_type,1>     pointers;
        jblib::Array<size_type,2>     coords;
        std::vector<_Tp>       val;

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
double SparseMatrix4D<_Tp>::calculateMemory() {
    const double mb = (1024.0*1024.0);
    if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP || matrixFormat == SPARSE_MATRIX_FORMAT_COO){
        return ((4*nnz*sizeof(size_type))+(val.size()*sizeof(_Tp)))/mb;
    } else if(matrixFormat == SPARSE_MATRIX_FORMAT_CSR){
        return (((3*nnz+dim[0]+1)*sizeof(size_type))+(val.size()*sizeof(_Tp)))/mb;
    }else{
        return ((4*nnz*sizeof(size_type))+(val.size()*sizeof(_Tp)))/mb;
    }
}

template <typename _Tp>
void SparseMatrix4D<_Tp>::insertValue(size_type i, size_type j, size_type k, size_type l, _Tp value) {

    if(matrixFormat == SPARSE_MATRIX_FORMAT_MAP) { // can only insert elements into map formatted matrix

        if(  !( ((i < dim[0]) && (i >= 0)) && ((j < dim[1]) && (j >= 0)) 
                    && ((k < dim[2]) && (k >= 0)) && ((l < dim[3]) && (l >= 0)) ) ) { // element must be inside matrix boundaries
            jams_error("Attempted to insert matrix element outside of matrix size");
        }

        i_idx.push_back(i);
        j_idx.push_back(j);
        k_idx.push_back(k);
        l_idx.push_back(l);

        val.push_back(value);

        nnz_unmerged++;
    } else {
        jams_error("Can only insert into MAP format sparse matrix");
    }

}

    template <typename _Tp>
void SparseMatrix4D<_Tp>::convertMAP2COO()
{

    nnz = 0;
    coords.resize(nnz_unmerged,4);

    if(nnz_unmerged > 0){
        for(int i=0; i<nnz_unmerged; ++i){
            coords(i,0) = i_idx[i];
            coords(i,1) = j_idx[i];
            coords(i,2) = k_idx[i];
            coords(i,3) = l_idx[i];
        }
    }

    nnz = nnz_unmerged;

    i_idx.clear();
    j_idx.clear();
    k_idx.clear();
    l_idx.clear();

    matrixFormat = SPARSE_MATRIX_FORMAT_COO;
}

    template <typename _Tp>
void SparseMatrix4D<_Tp>::convertMAP2CSR()
{

    // Upper most dimension is ordered (as pointers) but other dimensions
    // are not ordered

    pointers.resize(dim[0]+1);
    coords.resize(nnz_unmerged,3);

    std::vector<_Tp> csrval(nnz_unmerged,0);

    if(nnz_unmerged > 0){
        size_type count=0;
        for(int i=0;i<dim[0];++i){
            pointers(i) = count;

            for(int n=0;n<nnz_unmerged;++n){
                if( i_idx[n] == i ){
                    //std::cerr<< i << "\t" << count << "\t" << j_idx[n] << "\t" << k_idx[n] << "\t" << l_idx[n] << std::endl;
                    coords(count,0) = j_idx[n];
                    coords(count,1) = k_idx[n];
                    coords(count,2) = l_idx[n];
                    csrval[count] = val[n];
                    count++;
                }
            }
        }
        pointers(dim[0]) = count; // end pointer

        val.clear();
        val = csrval;

        nnz = count; assert(count == nnz_unmerged);
    }

    i_idx.clear();
    j_idx.clear();
    k_idx.clear();
    l_idx.clear();

    matrixFormat = SPARSE_MATRIX_FORMAT_CSR;
}

#endif // __SPARSEMATRIX_H__
