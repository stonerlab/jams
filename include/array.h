#ifndef __ARRAY_H__
#define __ARRAY_H__

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
    ~Array(){data.clear();}

    inline void clear() {
      dim0=0;
      data.clear();
    }

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


template <>
class Array<fftw_complex> 
{
  public:
    typedef unsigned int size_type;
    typedef fftw_complex value_type;
    typedef fftw_complex* iterator;
    typedef const fftw_complex* const_iterator;
    typedef ptrdiff_t difference_type;

    Array() : dim0(0), data(NULL) {}
    Array(size_type n) {resize(n);}
    ~Array(){clear();}

    inline void clear() {
      dim0=0;
      fftw_free(data);
      data=NULL;
    }

    void resize(size_type n) {
      dim0 = n;
      if(data != NULL){
          fftw_free(data);
      }
      data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
    }

    inline fftw_complex& RESTRICT operator()(const size_type i) {
      assert( i < dim0 );
      return data[i];
    }

    inline const fftw_complex& operator()(const size_type i) const {
      assert( i < dim0 );
      return data[i];
    }
    
    inline fftw_complex& RESTRICT operator[](const size_type i) {
      assert( i < dim0 );
      return data[i];
    }

    inline const fftw_complex& operator[](const size_type i) const {
      assert( i < dim0 );
      return data[i];
    }
    
    inline fftw_complex* RESTRICT ptr() {
      return data;
    }


    inline size_type size() const { return dim0; }

  private:
    size_type dim0;
    fftw_complex* data;
};
#endif // __ARRAY_H__
