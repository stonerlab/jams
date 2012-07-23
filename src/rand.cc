#include <limits>
#include <cmath>
#include <cassert>
#include <stdint.h>

#include "globals.h"
#include "rand.h"

void Random::seed(const unsigned long &x) {
  if (x > ul_limit) {
    jams_error("Random seed is too large");
  }

  if (x == 0) {
    mwc_x = static_cast<uint32_t>(time(NULL));
  } else {
    mwc_x = x;
  }

  for (int i = 0; i < 4096; ++i) {
    cmwc_q[i] = mwc32();
  }

  cmwc_c = 123;
  cmwc_r = 4095;

  normal_logic=false;

  init_seed = x;
  initialised = true;
  
}

///
/// @brief random number from the distribution [0,1) 
///
double Random::uniform() {
  assert(initialised == true);
  return (static_cast<double>(cmwc4096())*norm_open);
}

///
/// @brief random number from the distribution (0,1)
///
double Random::uniform_open() {
  assert(initialised == true);
  uint32_t x;
 
  do {
    x = cmwc4096();
  } while ( x == static_cast<uint32_t>(0) );

  return (static_cast<double>(x)*norm_open);
}

///
/// @brief random number from the distribution [0,1]
///
double Random::uniform_closed() {
  assert(initialised == true);
  return (static_cast<double>(cmwc4096())*norm_closed);
}

///
/// \section Details
/// This function produces a properly uniform discrete integer distribution. In other
/// words it avoids the bias that some values are more likely than others if the base 
/// 32bit random integers are not exactly divisible by the range \f$ L=n-m+1 \f$. It 
/// works by rejecting the excess \f$ 2^{32} \mbox{ mod } L \f$ values that are 
/// generated. This has some overhead, but the value of \f$ 2^{32} \mbox{ mod } L \f$ 
/// is stored as it is likely to be reused.
///
/// \section Notes
/// This may seem overkill however in Monte-Carlo codes it may be important that the
/// sampling is very uniform. It is easier to be stricter now than to prove (if at 
/// all possible) that the more naive method would produce the same result.
///
/// Also note that the casting of floating point values to integers is still naive
/// because there can still only be \f$2^{32}\f$ floating point values generated from the
/// underlying integer generator.
/// 
/// \section License
/// This code is Copyright 2001-2008 Agner Fog (http://www.agner.org)\n
/// GNU General Public License http://www.gnu.org/licenses/gpl.html
///
/// @param[in] m minimum distribution value
/// @param[in] n maximum distribution value 
/// @return integer random number from the distribution [n,m]
///
int Random::uniform_discrete(const int m, const int n) {
  assert(initialised == true);
  if (n < m) {
    jams_error("n must be > m in discrete uniform generator");
  }

  uint32_t  interval;   // Length of interval
  uint64_t  longran;    // Random bits * interval
  uint32_t  iran;       // Longran / 2^32
  uint32_t  remainder;  // Longran % 2^32

  interval = static_cast<uint32_t>(n - m + 1);
  if (interval != discrete_ival) {
    discrete_rlim = uint32_t((static_cast<uint64_t>(1) << 32) / interval) * interval - 1;
    discrete_ival = interval;
  }
  do { // Rejection loop
    longran   = static_cast<uint64_t>(cmwc4096() * interval);
    iran      = static_cast<uint32_t>(longran >> 32);
    remainder = static_cast<uint32_t>(longran);
  } while (remainder > discrete_rlim);

  return (static_cast<int32_t>(iran + m));
}

///
/// \section Details
/// This member uses the Marsaglia polar method to generate a pair of normally
/// distributed random variables. 
/// 
/// The mathematics is essentially the same as the Box-Muller algorithm, but avoids the
/// expensive sin and cos calls by rejection sampling of two random variables to lie
/// within the unit circle.
/// 
/// The rejection sampling of the two variables \f$ -1 < (x,y) < 1\f$ must give:
/// \f[ s = x^2 + y^2 < 1 \f]
/// Then the two normally distributed variables are:
/// \f[ x \sqrt{-2\ln(s)/s}, \quad y \sqrt{-2\ln(s)/s} \f]
///
/// One of the variables is returned immediately, the other is stored and returned on
/// the next call to save the generation of a new value.
///
/// @return standard normally distributed variable
///
double Random::normal() {
  assert(initialised == true);
  double s,x,y;
  if (normal_logic == true) {

    normal_logic = false;
    return normal_next;

  } else {
    do {
      x = -1.0 + static_cast<double>(cmwc4096())*norm_open2;
      y = -1.0 + static_cast<double>(cmwc4096())*norm_open2;

      s = (x*x) + (y*y);
      // floating point comparison below is needed to avoid log(0.0)
    } while (s > 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);

    normal_next = s*y;
    normal_logic = true;

    return s*x;
  }
}

void Random::sphere(double &x, double &y, double &z) {
    assert(initialised == true);
    double v1,v2,s,ss;

    do {
        v1 = -1.0 + static_cast<double>(cmwc4096())*norm_open2;
        v2 = -1.0 + static_cast<double>(cmwc4096())*norm_open2;
        s = (v1*v1) + (v2*v2);
    } while ( s > 1.0 );

    ss = sqrt(1.0 - s);

    x = 2.0*v1*ss;
    y = 2.0*v2*ss;
    z = 1.0 - 2.0*s;
}
