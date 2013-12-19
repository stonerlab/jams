#include "summations.h"

float64 jblib::KahanSum(const float64 * JB_RESTRICT data, const uint32 size){
  JB_REGISTER uint32 i;
  float64 y,t;
  float64 sum = 0.0;
  float64 c = 0.0; // a running compensation for lost low-order bits
  for( i=0; i!=size; ++i){
    y = data[i] - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}
