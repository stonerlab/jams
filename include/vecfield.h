#ifndef __VECFIELD_H__
#define __VECFIELD_H__

#include "globals.h"

#include <valarray>
#include <cassert>

template <typename _Tp>
class VecField
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;

    VecField(){ dim=0; }
    VecField(size_type n) {
      dim = 3*n;
      data.resize(dim);
    }

    void resize(size_type n) {
      dim = 3*n;
      data.resize(dim);
    }

    inline _Tp& RESTRICT x(const size_type i) {
      assert( (i >= 0) && (i < dim) );
      return data[i];
    }
    inline _Tp& RESTRICT y(const size_type i) {
      assert( (i >= 0) && (i < dim) );
      return data[i+dim];
    }
    inline _Tp& RESTRICT z(const size_type i) {
      assert( (i >= 0) && (i < dim) );
      return data[i+2*dim];
    }

    inline const _Tp& x(const size_type i) const {
      assert( (i >= 0) && (i < dim) );
      return data[i];
    }
    inline const _Tp& y(const size_type i) const {
      assert( (i >= 0) && (i < dim) );
      return data[i+dim];
    }
    inline const _Tp& z(const size_type i) const {
      assert( (i >= 0) && (i < dim) );
      return data[i+2*dim];
    }



    inline size_type size() const { return dim; }

  private:
    size_type dim;
    std::valarray<_Tp> data;
};

#endif // __VECFIELD_H__
