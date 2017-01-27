// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_VEC2_H
#define JBLIB_CONTAINERS_VEC2_H

#include <ostream>
#include <cmath>
#include <algorithm>

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

#include "jblib/math/equalities.h"

namespace jblib {
  template <typename type>
    class Vec2 {
      public:
        type x;
        type y;

        // Don't give default arguments of zero because this could be miss
        // interpreted if just one argument is given
        // default constructors
        Vec2() : x(0), y(0) {}
        Vec2(const type ix, const type iy) : x(ix), y(iy) {}
        Vec2(const Vec2<type> &other) : x(other.x), y(other.y) {}

        void set(const type ix, const type iy);

        type	    operator[](const int32 i) const;
        type &	    operator[](const int32 i);
        Vec2<type>	operator-() const;
        //Vec2<type> &	operator=( const Vec2 &a );
        Vec2<type>	operator*(const type a) const;
        Vec2<type>	operator/(const type a) const;
        Vec2<type>	operator+(const Vec2 &a) const;
        Vec2<type>	operator-(const Vec2 &a) const;
        Vec2<type> &	operator+=(const Vec2 &a);
        Vec2<type> &	operator-=(const Vec2 &a);
        Vec2<type> &	operator/=(const double a);
        Vec2<type> &	operator*=(const double a);

        template< typename ftype >
          friend void swap( const Vec2<ftype> &a, const Vec2<ftype> &b);

        template< typename ftype >
          friend type dot( const Vec2<ftype> &a, const Vec2<ftype> &b);
        template< typename ftype >
          friend Vec2<type>& cross( const Vec2<ftype> &a, const Vec2<ftype> &b);
        template< typename ftype >
          friend double abs( const Vec2<ftype> &a);
        template< typename ftype >
          friend double angle( const Vec2<ftype> &a, const Vec2<ftype> &b);

        template< typename ftype >
          friend std::ostream& operator<<(std::ostream& os, const Vec2<ftype> &vec);
    };

  template< typename type >
    inline void Vec2<type>::set(const type ix, const type iy){
      x = ix; y = iy;
    }

  template< typename type >
    inline type Vec2<type>::operator[]( const int32 i ) const{
      return (&x)[i];
    }

  template< typename type >
    inline type & Vec2<type>::operator[]( const int32 i ){
      return (&x)[i];
    }

  template< typename type >
    inline Vec2<type> Vec2<type>::operator-() const {
      return Vec2<type>( -x, -y);
    }

  template< typename type >
    inline Vec2<type> Vec2<type>::operator*(const type a) const{
      return Vec2<type>( a*x, a*y);
    }

  template< typename type >
    inline Vec2<type> Vec2<type>::operator/(const type a) const{
      const double norm = 1.0/a;
      return Vec2<type>(norm*x, norm*y);
    }

  template< typename type >
    inline Vec2<type> Vec2<type>::operator+(const Vec2<type> &a) const{
      return Vec2<type>( x+a.x, y+a.y);
    }

  template< typename type >
    inline Vec2<type> Vec2<type>::operator-(const Vec2<type> &a) const{
      return Vec2<type>( x-a.x, y-a.y);
    }

  template< typename type >
    inline Vec2<type> &Vec2<type>::operator+=(const Vec2<type> &a) {
      x += a.x; y += a.y;
      return *this;
    }

  template< typename type >
    inline Vec2<type> &Vec2<type>::operator-=(const Vec2<type> &a) {
      x -= a.x; y -= a.y;
      return *this;
    }

  template< typename type >
    inline Vec2<type> &Vec2<type>::operator/=(const double a) {
      x /= a; y /= a;
      return *this;
    }

  template< typename type >
    inline Vec2<type> &Vec2<type>::operator*=(const double a) {
      x *= a; y *= a;
      return *this;
    }

  // Friends
  template< typename type >
    inline void swap( const Vec2<type> &a, const Vec2<type> &b){
      std::swap(a.x,b.x); std::swap(a.y,b.y);
    }

  template< typename type >
    inline type dot( const Vec2<type> &a, const Vec2<type> &b){
      return (a.x*b.x + a.y*b.y);
    }

  template< typename type >
    inline double abs( const Vec2<type> &a){
      return sqrt( dot(a,a) );
    }

  template< typename type >
    inline double angle( const Vec2<type> &a, const Vec2<type> &b){
      return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    }

  template< typename type >
    std::ostream& operator<<(std::ostream& os, const Vec2<type> &vec) {
      return os << vec.x << "\t" << vec.y;
    }
}

#endif  // JBLIB_CONTAINERS_VEC2_H
