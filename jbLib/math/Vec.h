#ifndef JB_VEC_H
#define JB_VEC_H

// TODO: remove output to another header
#include <ostream>
#include <cmath>
#include <algorithm>

#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"
#include "../math/Math.h"


namespace jbLib {

  template< typename type >
    class Vec{

      public:
        type x;
        type y;
        type z;

        // Don't give default arguments of zero because this could be miss
        // interpreted if just one argument is given
        // default constructors
        Vec() : x(0), y(0), z(0) {}
        Vec(const type ix, const type iy, const type iz) : x(ix), y(iy), z(iz) {}
        Vec( const Vec<type> &other ) : x(other.x), y(other.y), z(other.z) {}

        void set(const type ix, const type iy, const type iz);

        type	    operator[]( const int32 i ) const;
        type &	    operator[]( const int32 i );
        Vec<type>	operator-() const;
        //Vec<type> &	operator=( const Vec &a );
        Vec<type>	operator*( const type a ) const;
        Vec<type>	operator/( const type a ) const;
        Vec<type>	operator+( const Vec &a ) const;
        Vec<type>	operator-( const Vec &a ) const;
        Vec<type> &	operator+=( const Vec &a );
        Vec<type> &	operator-=( const Vec &a );
        Vec<type> &	operator/=( const double a );
        Vec<type> &	operator*=( const double a );



        template< typename ftype >
          friend void swap( const Vec<ftype> &a, const Vec<ftype> &b); 

        template< typename ftype >
          friend type dot( const Vec<ftype> &a, const Vec<ftype> &b);
        template< typename ftype >
          friend Vec<type>& cross( const Vec<ftype> &a, const Vec<ftype> &b); 
        template< typename ftype >
          friend double abs( const Vec<ftype> &a);
        template< typename ftype >
          friend double angle( const Vec<ftype> &a, const Vec<ftype> &b); 

        template< typename ftype >
          friend std::ostream& operator<<(std::ostream& os, const Vec<ftype> &vec);

        //Vec<type>& operator=(Vec<type> rhs);

    };

  template< typename type >
    JB_INLINE void Vec<type>::set(const type ix, const type iy, const type iz){
      x = ix; y = iy; z = iz;
    }

  template< typename type >
    JB_INLINE type Vec<type>::operator[]( const int32 i ) const{
      return (&x)[i];
    }

  template< typename type >
    JB_INLINE type & Vec<type>::operator[]( const int32 i ){
      return (&x)[i];
    }

  template< typename type >
    JB_INLINE Vec<type> Vec<type>::operator-() const {
      return Vec<type>( -x, -y, -z);
    }

  //template< typename type >
  //JB_INLINE Vec<type> & Vec<type>::operator=(const Vec<type> &a) {
  //x = a.x; y = a.y, z = a.z;
  //return *this;
  //}

  template< typename type >
    JB_INLINE Vec<type> Vec<type>::operator*(const type a) const{
      return Vec<type>( a*x, a*y, a*z );
    }

  template< typename type >
    JB_INLINE Vec<type> Vec<type>::operator/(const type a) const{
      const double norm = 1.0/a;
      return Vec<type>( norm*x, norm*y, norm*z );
    }

  template< typename type >
    JB_INLINE Vec<type> Vec<type>::operator+(const Vec<type> &a) const{
      return Vec<type>( x+a.x, y+a.y, z+a.z );
    }

  template< typename type >
    JB_INLINE Vec<type> Vec<type>::operator-(const Vec<type> &a) const{
      return Vec<type>( x-a.x, y-a.y, z-a.z );
    }

  template< typename type >
    JB_INLINE Vec<type> &Vec<type>::operator+=(const Vec<type> &a) {
      x += a.x; y += a.y; z += a.z;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec<type> &Vec<type>::operator-=(const Vec<type> &a) {
      x -= a.x; y -= a.y; z -= a.z;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec<type> &Vec<type>::operator/=(const double a) {
      x /= a; y /= a; z /= a;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec<type> &Vec<type>::operator*=(const double a) {
      x *= a; y *= a; z *= a;
      return *this;
    }





  //template< typename type >
  //JB_INLINE Vec<type>& Vec<type>::operator=(Vec<type> rhs){
  //swap(*this, rhs);
  //return *this;
  //}

  // Friends
  template< typename type >
    JB_INLINE void swap( const Vec<type> &a, const Vec<type> &b){ 
      std::swap(a.x,b.x); std::swap(a.y,b.y); std::swap(a.z,b.z);
    }

  template< typename type >
    JB_INLINE type dot( const Vec<type> &a, const Vec<type> &b){ 
      return (a.x*b.x + a.y*b.y + a.z*b.z);
    }

  template< typename type >
    JB_INLINE Vec<type>& cross( const Vec<type> &a, const Vec<type> &b){ 
      return Vec<type>( (a.y*b.z-a.z*b.y), (a.z*b.x-a.x*b.z), (a.x*b.y - a.y*b.x) );
    }

  template< typename type >
    JB_INLINE double abs( const Vec<type> &a){ 
      return sqrt( dot(a,a) );
    }

  template< typename type >
    JB_INLINE double angle( const Vec<type> &a, const Vec<type> &b){ 
      return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    }


  ///
  /// @brief returns an integer count of the number of zero elements in (x,y,z)
  ///
  /// @note bool to int conversion is defined in the C++ standard. 
  ///       See http://stackoverflow.com/questions/5369770/bool-to-int-conversion
  ///
  JB_INLINE int vecCountZeros(const Vec<double> &a){
    using namespace jbMath;

    return ( floatEquality(a.x,0.0) + floatEquality(a.y,0.0) + floatEquality(a.z,0.0) );
  }

  template< typename type >
    std::ostream& operator<<(std::ostream& os, const Vec<type> &vec)
    {
      return os << vec.x << "\t" << vec.y << "\t" << vec.z;
    }

}
#endif
