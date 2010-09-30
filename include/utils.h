#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <sstream>
#include <locale>
#include <functional>
#include <algorithm>
#include <cmath>

#include "consts.h"

inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
inline std::string &rtrim(std::string &s) {
s.erase(std::find_if(s.rbegin(), s.rend(),std::not1(std::ptr_fun<int,int>(std::isspace))).base(), s.end());
return s;
}

// trim from both ends
inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}

inline double deg_to_rad(const double &angle) {
  return angle*(pi/180.0);
}

inline double rad_to_deg(const double &angle) {
  return angle*(180.0/pi);
}

///
/// @brief  Sign transfer function from y->x
///
/// The sign transfer function transfers the sign of argument y to the
/// value of x. This is defined as for the Fortran function of the same
/// name.
///
/// @param[in]  x value to transfer to
/// @param[in]  y value to take sign from 
/// @return x with the sign of y
///
template <typename _Tp1, typename _Tp2>
inline _Tp1 jams::math::sign(const _Tp1 &x, const _Tp2 &y)
{
  if(y >= 0.0)
  {
    return std::abs(x);
  } else {
    return -std::abs(x);
  }
}

///
/// @brief  Generates next point space symmetry representation
///
/// This function generates the next point space symmetry representation from
/// a set of points, where the ascending positive valued points are the lowest
/// configuration and the negative descending the highest.
///
/// @param  [in,out] pts[3] points in space
/// @return false if no more symmetry points can be generated, otherwise true
///
template <typename _Tp>
bool next_point_symmetry(_Tp pts[3]) {
  // check number is valid (0 cannot be negative)
  bool valid = false;

  // extract sign mask
  int sgn[3] = { sign(1,pts[0]), sign(1,pts[1]), sign(1,pts[2]) };

  // absolute value of the points
  _Tp abspts[3] = { std::abs(pts[0]), std::abs(pts[1]), std::abs(pts[2])};

  // loop until number are valied wrt 0
  do{
    // permute the absolute values
    if(std::next_permutation(abspts,abspts+3)){
      pts[0] = abspts[0]*sgn[0];
      pts[1] = abspts[1]*sgn[1];
      pts[2] = abspts[2]*sgn[2];

    // if no more permutation is possible then permute the signs
    }else{

      // once -1,-1,-1 is reached, all sgnmetries have been computed
      if(sgn[0]+sgn[1]+sgn[2] == -3){
        return false;
      }

      // re-sort absolute values to ascend for permutation
      std::sort(abspts,abspts+3);

      // permute signs if possible
      if(std::next_permutation(sgn,sgn+3)){
        pts[0] = abspts[0]*sgn[0];
        pts[1] = abspts[1]*sgn[1];
        pts[2] = abspts[2]*sgn[2];
      }else{

        // sort signs so that sgn[2] is 1 if possible
        std::sort(sgn,sgn+3);
        sgn[2] = -1;
        // re-sort so that numbers are ascending for permutation
        std::sort(sgn,sgn+3);

        pts[0] = abspts[0]*sgn[0];
        pts[1] = abspts[1]*sgn[1];
        pts[2] = abspts[2]*sgn[2];
      }
    }

    // check validity of values with respect to zero signs
    valid = true;
    for(int i=0;i<3;++i){
      if(pts[i] == 0 && sgn[i] == -1){
        valid = false;
      }
    }
  } while (valid==false);

  return true;
}

inline void jams::math::cartesian_to_spherical(const double x,
    const double y, const double z, double& r, double& theta, double& phi)
{
  r       = sqrt(x*x+y*y+z*z);
  theta   = acos( z/r );
  phi     = atan2(y,x);
}

inline void jams::math::spherical_to_cartesian(const double r,
    const double theta, const double phi, double& x, double& y, double& z)
{
  x = r*cos(theta)*cos(phi);
  y = r*cos(theta)*sin(phi);
  z = r*sin(theta);
}
i

#endif // __UTILS_H__
