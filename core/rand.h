class Random;

#ifndef JAMS_CORE_RAND_H
#define JAMS_CORE_RAND_H

#include <limits>
#include <stdint.h>
#include <vector>

class Random 
{
  public:
    Random() : 
      initialised(false),
      init_seed(0),
      ul_limit(std::numeric_limits<uint32_t>::max()),
      norm_open(1.0/static_cast<double>(ul_limit)),
      norm_open2(2.0/static_cast<double>(ul_limit)),
      norm_closed(1.0/static_cast<double>(ul_limit-1)),
      mwc_x(0),
      cmwc_q(4096,0),
      cmwc_c(0),
      cmwc_r(0),
      discrete_ival(0),
      discrete_rlim(0),
      normal_logic(false),
      normal_next(0.0)
    {}

    void seed(const unsigned long &x);
    double uniform();
    double uniform_open();
    double uniform_closed();
    int    uniform_discrete(const int m, const int n);
    double normal();
    void   sphere(double &x, double &y, double &z);
  private:
    bool initialised;
    unsigned long init_seed;

    const uint32_t ul_limit;
    const double norm_open;
    const double norm_open2;
    const double norm_closed;

    uint64_t mwc_x;

    std::vector<uint32_t> cmwc_q;
    uint32_t cmwc_c;
    uint32_t cmwc_r;

    uint32_t        discrete_ival;
    uint32_t        discrete_rlim;

    bool normal_logic;
    double normal_next;
    
    uint32_t mwc32() {
      return mwc_x = ( (mwc_x&static_cast<uint64_t>(0xffffffff))*(static_cast<uint32_t>(4294967118U))+(mwc_x>>32) );
    }

    uint32_t cmwc4096() {
//      const uint64_t a = static_cast<uint64_t>(18782);
//      const uint32_t m = static_cast<uint32_t>(0xfffffffe);
      uint64_t t;
      uint32_t x;

      cmwc_r = ( cmwc_r + 1 ) & 4095;
      t = static_cast<uint64_t>(18782) * cmwc_q[cmwc_r] + cmwc_c;
      cmwc_c = (t >> 32);
      x = t + cmwc_c;
      if( x < cmwc_c ){ cmwc_r++; cmwc_c++; }
      return ( cmwc_q[cmwc_r] = (static_cast<uint32_t>(0xfffffffe) - x) );
    }
    
};

#endif // JAMS_CORE_RAND_H
