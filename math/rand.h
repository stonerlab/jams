class Random;

#ifndef JB_RAND_H
#define JB_RAND_H

#include <limits>
#include <stdint.h>
#include <vector>

#include "../sys/defines.h"
#include "../sys/types.h"

class Random 
{
  public:
    Random(); 

    void    seed(const uint64 &x);

    float64 uniform();
    float64 uniform_open();
    float64 uniform_closed();
    int32   uniform_discrete(const int32 m, const int32 n);

    float64 normal();

    void    sphere(float64 &x, float64 &y, float64 &z);
  private:
    bool    initialised;
    uint64  init_seed;

    const uint32_t ul_limit;
    const float64 norm_open;
    const float64 norm_open2;
    const float64 norm_closed;

    uint64 mwc_x;

    std::vector<uint32> cmwc_q;
    uint32 cmwc_c;
    uint32 cmwc_r;

    uint32  discrete_ival;
    uint32  discrete_rlim;

    bool    normal_logic;
    float64 normal_next;
    
    uint32  mwc32();
    uint32  cmwc4096();
    
};

Random::Random() : 
  initialised(false),
  init_seed(0),
  ul_limit(std::numeric_limits<uint32>::max()),
  norm_open(1.0/static_cast<float64>(ul_limit)),
  norm_open2(2.0/static_cast<float64>(ul_limit)),
  norm_closed(1.0/static_cast<float64>(ul_limit-1)),
  mwc_x(0),
  cmwc_q(4096,0),
  cmwc_c(0),
  cmwc_r(0),
  discrete_ival(0),
  discrete_rlim(0),
  normal_logic(false),
  normal_next(0.0)
{}

inline uint32 Random::mwc32() {
  return mwc_x = ( (mwc_x&(0xffffffffULL))*(4294967118ULL)+(mwc_x>>32) );
}

inline uint32 Random::cmwc4096() {
  uint64 t;
  uint32 x;

  cmwc_r = ( cmwc_r + 1 ) & 4095;
  t = (18782ULL) * cmwc_q[cmwc_r] + cmwc_c;
  cmwc_c = (t >> 32);
  x = t + cmwc_c;
  if( x < cmwc_c ){ cmwc_r++; cmwc_c++; }
  return ( cmwc_q[cmwc_r] = ((0xfffffffeULL) - x) );
}
#endif // JB_RAND_H
