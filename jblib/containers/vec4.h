#ifndef JB_MATH_VEC4_H
#define JB_MATH_VEC4_H

#include <cmath>
#include <algorithm>

#include "../sys/defines.h"
#include "../sys/types.h"

#include "../math/equalities.h"


namespace jblib {

  template< typename valueType_ >
    class Vec4{

      public:
        valueType_ x;
        valueType_ y;
        valueType_ z;
        valueType_ w;

        // Don't give default arguments of zero because this could be miss
        // interpreted if just one argument is given
        // default constructors
        Vec4() : x(0), y(0), z(0), w(0) {}
        Vec4(const valueType_ ix, const valueType_ iy, const valueType_ iz, const valueType_ iw) : x(ix), y(iy), z(iz), w(iw) {}
        Vec4( const Vec4<valueType_> &other ) : x(other.x), y(other.y), z(other.z), w(other.w) {}

        void set(const valueType_ ix, const valueType_ iy, const valueType_ iz, const valueType_ iw);

        valueType_	    operator[]( const int32 i ) const;
        valueType_ &	    operator[]( const int32 i );
        Vec4<valueType_>	operator-() const;
        //Vec4<valueType_> &	operator=( const Vec4 &a );
        Vec4<valueType_>	operator*( const valueType_ a ) const;
        Vec4<valueType_>	operator/( const valueType_ a ) const;
        Vec4<valueType_>	operator+( const Vec4 &a ) const;
        Vec4<valueType_>	operator-( const Vec4 &a ) const;
        Vec4<valueType_> &	operator+=( const Vec4 &a );
        Vec4<valueType_> &	operator-=( const Vec4 &a );
        Vec4<valueType_> &	operator/=( const double a );
        Vec4<valueType_> &	operator*=( const double a );

        bool operator==(const Vec4 &a);


        template< typename fvalueType_ >
          friend void swap( const Vec4<fvalueType_> &a, const Vec4<fvalueType_> &b); 

        template< typename fvalueType_ >
          friend valueType_ dot( const Vec4<fvalueType_> &a, const Vec4<fvalueType_> &b);
        template< typename fvalueType_ >
          friend Vec4<valueType_>& cross( const Vec4<fvalueType_> &a, const Vec4<fvalueType_> &b); 
        template< typename fvalueType_ >
          friend double abs( const Vec4<fvalueType_> &a);
        template< typename fvalueType_ >
          friend double angle( const Vec4<fvalueType_> &a, const Vec4<fvalueType_> &b); 

        template< typename fvalueType_ >
          friend std::ostream& operator<<(std::ostream& os, const Vec4<fvalueType_> &vec);

        //Vec4<valueType_>& operator=(Vec4<valueType_> rhs);

    };

  template< typename valueType_ >
    JB_INLINE void Vec4<valueType_>::set(const valueType_ ix, const valueType_ iy, const valueType_ iz, const valueType_ iw){
      x = ix; y = iy; z = iz; w = iw;
    }

  template< typename valueType_ >
    JB_INLINE valueType_ Vec4<valueType_>::operator[]( const int32 i ) const{
      return (&x)[i];
    }

  template< typename valueType_ >
    JB_INLINE valueType_ & Vec4<valueType_>::operator[]( const int32 i ){
      return (&x)[i];
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> Vec4<valueType_>::operator-() const {
      return Vec4<valueType_>( -x, -y, -z, -w);
    }

  //template< typename valueType_ >
  //JB_INLINE Vec4<valueType_> & Vec4<valueType_>::operator=(const Vec4<valueType_> &a) {
  //x = a.x; y = a.y, z = a.z;
  //return *this;
  //}

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> Vec4<valueType_>::operator*(const valueType_ a) const{
      return Vec4<valueType_>( a*x, a*y, a*z, a*w );
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> Vec4<valueType_>::operator/(const valueType_ a) const{
      const double norm = 1.0/a;
      return Vec4<valueType_>( norm*x, norm*y, norm*z, norm*w);
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> Vec4<valueType_>::operator+(const Vec4<valueType_> &a) const{
      return Vec4<valueType_>( x+a.x, y+a.y, z+a.z, w+a.w);
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> Vec4<valueType_>::operator-(const Vec4<valueType_> &a) const{
      return Vec4<valueType_>( x-a.x, y-a.y, z-a.z, w-a.w);
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> &Vec4<valueType_>::operator+=(const Vec4<valueType_> &a) {
      x += a.x; y += a.y; z += a.z; w += a.w;
      return *this;
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> &Vec4<valueType_>::operator-=(const Vec4<valueType_> &a) {
      x -= a.x; y -= a.y; z -= a.z; w -= a.w;
      return *this;
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> &Vec4<valueType_>::operator/=(const double a) {
      x /= a; y /= a; z /= a; w /= a;
      return *this;
    }

  template< typename valueType_ >
    JB_INLINE Vec4<valueType_> &Vec4<valueType_>::operator*=(const double a) {
      x *= a; y *= a; z *= a; w *= a;
      return *this;
    }





  //template< typename valueType_ >
  //JB_INLINE Vec4<valueType_>& Vec4<valueType_>::operator=(Vec4<valueType_> rhs){
  //swap(*this, rhs);
  //return *this;
  //}

  // Friends
  template< typename valueType_ >
    JB_INLINE void swap( const Vec4<valueType_> &a, const Vec4<valueType_> &b){ 
      std::swap(a.x,b.x); std::swap(a.y,b.y); std::swap(a.z,b.z); std::swap(a.w,b.w);
    }

  template< typename valueType_ >
    JB_INLINE valueType_ dot( const Vec4<valueType_> &a, const Vec4<valueType_> &b){ 
      return (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w);
    }

  //template< typename valueType_ >
    //JB_INLINE Vec4<valueType_>& cross( const Vec4<valueType_> &a, const Vec4<valueType_> &b){ 
      //return Vec4<valueType_>( (a.y*b.z-a.z*b.y), (a.z*b.x-a.x*b.z), (a.x*b.y - a.y*b.x) );
    //}

  template< typename valueType_ >
    JB_INLINE double abs( const Vec4<valueType_> &a){ 
      return sqrt( dot(a,a) );
    }

  template< typename valueType_ >
  bool Vec4<valueType_>::operator==(const Vec4<valueType_> &a){
      return ( x == a.x ) && ( y == a.y ) && ( z == a.z ) && ( w == a.w );
  }

  //template< typename valueType_ >
    //JB_INLINE double angle( const Vec4<valueType_> &a, const Vec4<valueType_> &b){ 
      //return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    //}


  ///
  /// @brief returns an integer count of the number of zero elements in (x,y,z)
  ///
  /// @note bool to int conversion is defined in the C++ standard. 
  ///       See http://stackoverflow.com/questions/5369770/bool-to-int-conversion
  ///
  JB_INLINE int vecCountZeros(const Vec4<double> &a){
    return ( floatEquality(a.x,0.0) + floatEquality(a.y,0.0) + floatEquality(a.z,0.0) + floatEquality(a.w,0.0) );
  }

  template< typename valueType_ >
    std::ostream& operator<<(std::ostream& os, const Vec4<valueType_> &vec)
    {
      return os << vec.x << "\t" << vec.y << "\t" << vec.z << "\t" << vec.w;
    }

}
#endif
