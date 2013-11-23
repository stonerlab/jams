#ifndef __ARRAY5D_H__
#define __ARRAY5D_H__

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#include <vector>
#include <cassert>
#include <cstddef>
#include <fftw3.h>

template <typename _Tp>
class Array5D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array5D() : dim0(0), dim1(0), dim2(0), dim3(0), dim4(0), data(0) {}

    Array5D(size_type d0, size_type d1, size_type d2, size_type d3, size_type d4)
      : dim0(d0), dim1(d1), dim2(d2), dim3(d3), dim4(d4), data(d0*d1*d2*d3*d4) {}

    ~Array5D(){data.clear();}

    inline void clear() {
      dim0 = 0;
      dim1 = 0;
      dim2 = 0;
      dim3 = 0;
      dim4 = 0;
      data.clear();
    }

    inline void resize(const size_type d0, const size_type d1, const size_type d2, const size_type d3, const size_type d4) {
      dim0 = d0; dim1 = d1; dim2 = d2; dim3 = d3; dim4 = d4;
      data.resize(d0*d1*d2*d3*d4);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k, const size_type l, const size_type m) {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      assert( l < dim3 );
      assert( m < dim4 );
      return data[(((i*dim1+j)*dim2+k)*dim3+l)*dim4+m];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k, const size_type l, const size_type m) const {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      assert( l < dim3 );
      assert( m < dim4 );
      return data[(((i*dim1+j)*dim2+k)*dim3+l)*dim4+m];
    }

//    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim0;
    size_type dim1;
    size_type dim2;
    size_type dim3;
    size_type dim4;
    std::vector<_Tp> data;
};



#endif // __ARRAY5D_H__
