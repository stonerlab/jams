#ifndef __ARRAY_H__
#define __ARRAY_H__

#include "globals.h"

#include <vector>
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

    Array() : dim0(0), data(0) {}
    Array(size_type n) : dim0(n), data(dim0) {}

    void resize(size_type n) {
      dim0 = n;
      data.resize(dim0);
    }

    inline _Tp& RESTRICT operator()(const size_type i) {
      assert( i < dim0 );
      return data[i];
    }

    inline const _Tp& operator()(const size_type i) const {
      assert( i < dim0 );
      return data[i];
    }
    
    inline _Tp& RESTRICT operator[](const size_type i) {
      assert( i < dim0 );
      return data[i];
    }

    inline const _Tp& operator[](const size_type i) const {
      assert( i < dim0 );
      return data[i];
    }
    
    inline _Tp* RESTRICT ptr() {
      return &(data[0]);
    }


    inline size_type size() const { return dim0; }

  private:
    size_type dim0;
    std::vector<_Tp> data;
};

#endif // __ARRAY_H__
