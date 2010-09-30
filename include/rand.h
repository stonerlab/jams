class Random;

#ifndef __RAND_H__
#define __RAND_H__

#include <stdint.h>

class Random 
{
  public:
    void seed(const unsigned long &x);
    double uniform();
    double uniform_open();
    double uniform_closed();
    int    uniform_discrete(const int m, const int n);
    double normal();
  private:
    bool initialised;
    unsigned long init_seed;

    uint64_t mwc_x;

    uint32_t cmwc_q[4096];
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
      const uint64_t a = static_cast<uint64_t>(18782);
      const uint32_t m = static_cast<uint32_t>(0xfffffffe);
      uint64_t t;
      uint32_t x;

      cmwc_r = ( cmwc_r + 1 ) & 4095;
      t = a * cmwc_q[cmwc_r] + cmwc_c;
      cmwc_c = (t >> 32);
      x = t + cmwc_c;
      if( x < cmwc_c ){ cmwc_r++; cmwc_c++; }
      return ( cmwc_q[cmwc_r] = (m - x) );
    }
    
};

#endif // __RAND_H__
