#ifndef __ARRAY2D_H__
#define __ARRAY2D_H__

#include <vector>
#include <cassert>

#include "consts.h"


// C ordered array
template <typename _Tp>
class Array2D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array2D() : dim0(0), dim1(0), data(0) {}

    Array2D(size_type d0, size_type d1)
      : dim0(d0), dim1(d1), data(d0*d1) {}

    inline void resize(size_type d0, size_type d1) {
      dim0 = d0; dim1 = d1;
      data.resize(d0*d1);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j) {
      assert( i < dim0 );
      assert( j < dim1 );
      return data[i*dim1+j];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j) const {
      assert( i < dim0 );
      assert( j < dim1 );
      return data[i*dim1+j];
    }

//    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim0;
    size_type dim1;
    std::vector<_Tp> data;
};

#endif // __ARRAY2D_H__
