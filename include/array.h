#ifndef __ARRAY_H__
#define __ARRAY_H__

#include "globals.h"

#include <valarray>
#include <cassert>

template <typename _Tp>
class Array 
{
  public:
    typedef unsigned int size_type;
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef ptrdiff_t difference_type;

    Array() {dim = 0;}

    Array(size_type n) {
      dim = n;
      data.resize(dim);
    }

    void resize(size_type n) {
      dim = n;
      data.resize(dim);
    }

    inline _Tp& RESTRICT operator()(const size_type i) {
      assert( (i >= 0) && (i < dim) );
      return data[i];
    }

    inline const _Tp& operator()(const size_type i) const {
      assert( (i >= 0) && (i < dim) );
      return data[i];
    }


    inline size_type size() const { return dim; }

  private:
    size_type dim;
    std::valarray<_Tp> data;
};

#endif // __ARRAY_H__
