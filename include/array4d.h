#ifndef __ARRAY4D_H__
#define __ARRAY4D_H__

#include "globals.h"

#include <vector>
#include <cassert>

template <typename _Tp>
class Array4D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array4D() : dim0(0), dim1(0), dim2(0), dim3(0), data(0) {}

    Array4D(size_type d0, size_type d1, size_type d2, size_type d3)
      : dim0(d0), dim1(d1), dim2(d2), dim3(d3), data(d0*d1*d2*d3) {}

    ~Array4D(){data.clear();}

    inline void clear() {
      dim0 = 0;
      dim1 = 0;
      dim2 = 0;
      dim3 = 0;
      data.clear();
    }

    inline void resize(const size_type d0, const size_type d1, const size_type d2, const size_type d3) {
      dim0 = d0; dim1 = d1; dim2 = d2; dim3 = d3;
      data.resize(d0*d1*d2*d3);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k, const size_type l) {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      assert( l < dim3 );
      return data[((i*dim1+j)*dim2+k)*dim3+l];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k, const size_type l) const {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      assert( l < dim3 );
      return data[((i*dim1+j)*dim2+k)*dim3+l];
    }

//    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim0;
    size_type dim1;
    size_type dim2;
    size_type dim3;
    std::vector<_Tp> data;
};

#endif // __ARRAY4D_H__
