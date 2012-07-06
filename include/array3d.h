#ifndef __ARRAY3D_H__
#define __ARRAY3D_H__

#ifdef __GNUC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#include <vector>
#include <cassert>
#include <cstddef>

template <typename _Tp>
class Array3D 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array3D() : dim0(0), dim1(0), dim2(0), data(0) {}

    Array3D(size_type d0, size_type d1, size_type d2)
      : dim0(d0), dim1(d1), dim2(d2), data(d0*d1*d2) {}

    ~Array3D(){data.clear();}

    inline void clear() {
      dim0 = 0;
      dim1 = 0;
      dim2 = 0;
      data.clear();
    }

    inline void resize(size_type d0, size_type d1, size_type d2) {
      dim0 = d0; dim1 = d1; dim2 = d2;
      data.resize(d0*d1*d2);
    }

    inline _Tp& RESTRICT operator()(const size_type i, const size_type j,
        const size_type k) {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      return data[(i*dim1+j)*dim2+k];
    }
    
    inline const _Tp& operator()(const size_type i, const size_type j,
        const size_type k) const {
      assert( i < dim0 );
      assert( j < dim1 );
      assert( k < dim2 );
      return data[(i*dim1+j)*dim2+k];
    }

//    inline size_type size(const size_type i) const { return dim[i]; }

  private:
    size_type dim0;
    size_type dim1;
    size_type dim2;
    std::vector<_Tp> data;
};

#endif // __ARRAY3D_H__
