#include "Math.h"

namespace jbLib {
  double jbMath::TrivialSum( const double * JB_RESTRICT data, const uint32 size){
    JB_REGISTER uint32 i;
    double sum = 0.0;
    for( i=0; i<size; ++i) {
      sum+=data[i];
    }
    return sum;
  }


  double jbMath::KahanSum(const double * JB_RESTRICT data, const uint32 size){
    JB_REGISTER uint32 i;
    double y,t;
    double sum = 0.0;
    double c = 0.0; // a running compensation for lost low-order bits
    for( i=0; i<size; ++i){
      y = data[i] - c;
      t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }

    return sum;
  }
}
