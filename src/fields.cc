#include "fields.h"
#include "globals.h"
#include "sparsematrix.h"

void calculate_fields()
{
  using namespace globals;
 
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  const char transa[1] = {'N'};
  const char matdescra[6] = {'S','L','N','C','N','N'};
  int i,j;

  if(Jij.nonzero() > 0) {
    jams_dcsrmv(transa,nspins3,nspins3,1.0,matdescra,Jij.ptrVal(),
        Jij.ptrCol(), Jij.ptrB(),Jij.ptrE(),s.ptr(),0.0,h.ptr()); 
  } else {
    std::fill(h.ptr(),h.ptr()+nspins3,0.0); 
  }

  // normalize by the gyroscopic factor
  for(i=0; i<nspins; ++i) {
    for(j=0; j<3;++j) {
      h(i,j) = (h(i,j)+w(i,j)+h_app[j]*mus(i))*gyro(i);
    }
  }
}
