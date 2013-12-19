#ifndef JB_MATH_VEC3_H
#define JB_MATH_VEC3_H

// TODO: remove output to another header
#include <ostream>
#include <cmath>
#include <algorithm>

#include "../sys/defines.h"
#include "../sys/types.h"

#include "../math/equalities.h"


namespace jblib {

  template< typename type >
    class Vec3{

      public:
        type x;
        type y;
        type z;

        // Don't give default arguments of zero because this could be miss
        // interpreted if just one argument is given
        // default constructors
        Vec3() : x(0), y(0), z(0) {}
        Vec3(const type ix, const type iy, const type iz) : x(ix), y(iy), z(iz) {}
        Vec3( const Vec3<type> &other ) : x(other.x), y(other.y), z(other.z) {}

        void set(const type ix, const type iy, const type iz);

        type	    operator[]( const int32 i ) const;
        type &	    operator[]( const int32 i );
        Vec3<type>	operator-() const;
        //Vec3<type> &	operator=( const Vec3 &a );
        Vec3<type>	operator*( const type a ) const;
        Vec3<type>	operator/( const type a ) const;
        Vec3<type>	operator+( const Vec3 &a ) const;
        Vec3<type>	operator-( const Vec3 &a ) const;
        Vec3<type> &	operator+=( const Vec3 &a );
        Vec3<type> &	operator-=( const Vec3 &a );
        Vec3<type> &	operator/=( const double a );
        Vec3<type> &	operator*=( const double a );



        template< typename ftype >
          friend void swap( const Vec3<ftype> &a, const Vec3<ftype> &b); 

        template< typename ftype >
          friend type dot( const Vec3<ftype> &a, const Vec3<ftype> &b);
        template< typename ftype >
          friend Vec3<type>& cross( const Vec3<ftype> &a, const Vec3<ftype> &b); 
        template< typename ftype >
          friend double abs( const Vec3<ftype> &a);
        template< typename ftype >
          friend double angle( const Vec3<ftype> &a, const Vec3<ftype> &b); 

        template< typename ftype >
          friend std::ostream& operator<<(std::ostream& os, const Vec3<ftype> &vec);

        //Vec3<type>& operator=(Vec3<type> rhs);

    };

  template< typename type >
    JB_INLINE void Vec3<type>::set(const type ix, const type iy, const type iz){
      x = ix; y = iy; z = iz;
    }

  template< typename type >
    JB_INLINE type Vec3<type>::operator[]( const int32 i ) const{
      return (&x)[i];
    }

  template< typename type >
    JB_INLINE type & Vec3<type>::operator[]( const int32 i ){
      return (&x)[i];
    }

  template< typename type >
    JB_INLINE Vec3<type> Vec3<type>::operator-() const {
      return Vec3<type>( -x, -y, -z);
    }

  //template< typename type >
  //JB_INLINE Vec3<type> & Vec3<type>::operator=(const Vec3<type> &a) {
  //x = a.x; y = a.y, z = a.z;
  //return *this;
  //}

  template< typename type >
    JB_INLINE Vec3<type> Vec3<type>::operator*(const type a) const{
      return Vec3<type>( a*x, a*y, a*z );
    }

  template< typename type >
    JB_INLINE Vec3<type> Vec3<type>::operator/(const type a) const{
      const double norm = 1.0/a;
      return Vec3<type>( norm*x, norm*y, norm*z );
    }

  template< typename type >
    JB_INLINE Vec3<type> Vec3<type>::operator+(const Vec3<type> &a) const{
      return Vec3<type>( x+a.x, y+a.y, z+a.z );
    }

  template< typename type >
    JB_INLINE Vec3<type> Vec3<type>::operator-(const Vec3<type> &a) const{
      return Vec3<type>( x-a.x, y-a.y, z-a.z );
    }

  template< typename type >
    JB_INLINE Vec3<type> &Vec3<type>::operator+=(const Vec3<type> &a) {
      x += a.x; y += a.y; z += a.z;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec3<type> &Vec3<type>::operator-=(const Vec3<type> &a) {
      x -= a.x; y -= a.y; z -= a.z;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec3<type> &Vec3<type>::operator/=(const double a) {
      x /= a; y /= a; z /= a;
      return *this;
    }

  template< typename type >
    JB_INLINE Vec3<type> &Vec3<type>::operator*=(const double a) {
      x *= a; y *= a; z *= a;
      return *this;
    }





  //template< typename type >
  //JB_INLINE Vec3<type>& Vec3<type>::operator=(Vec3<type> rhs){
  //swap(*this, rhs);
  //return *this;
  //}

  // Friends
  template< typename type >
    JB_INLINE void swap( const Vec3<type> &a, const Vec3<type> &b){ 
      std::swap(a.x,b.x); std::swap(a.y,b.y); std::swap(a.z,b.z);
    }

  template< typename type >
    JB_INLINE type dot( const Vec3<type> &a, const Vec3<type> &b){ 
      return (a.x*b.x + a.y*b.y + a.z*b.z);
    }

  template< typename type >
    JB_INLINE Vec3<type>& cross( const Vec3<type> &a, const Vec3<type> &b){ 
      return Vec3<type>( (a.y*b.z-a.z*b.y), (a.z*b.x-a.x*b.z), (a.x*b.y - a.y*b.x) );
    }

  template< typename type >
    JB_INLINE double abs( const Vec3<type> &a){ 
      return sqrt( dot(a,a) );
    }

  template< typename type >
    JB_INLINE double angle( const Vec3<type> &a, const Vec3<type> &b){ 
      return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    }


  ///
  /// @brief returns an integer count of the number of zero elements in (x,y,z)
  ///
  /// @note bool to int conversion is defined in the C++ standard. 
  ///       See http://stackoverflow.com/questions/5369770/bool-to-int-conversion
  ///
  JB_INLINE int vecCountZeros(const Vec3<double> &a){
    return ( floatEquality(a.x,0.0) + floatEquality(a.y,0.0) + floatEquality(a.z,0.0) );
  }

  template< typename type >
    std::ostream& operator<<(std::ostream& os, const Vec3<type> &vec)
    {
      return os << vec.x << "\t" << vec.y << "\t" << vec.z;
    }

}
#endif
