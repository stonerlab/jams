#include "fields.h"
#include "globals.h"
#include "sparsematrix.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

void calculate_fields()
{
  using namespace globals;
 
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  char transa[1] = {'N'};
  char matdescra[6] = {'S','L','N','C','N','N'};
  double one=1.0;
  double zero=0.0;
  int i,j;

  if(Jij.nonzero() > 0) {
#ifdef MKL
    mkl_dcsrmv(transa,&nspins3,&nspins3,&one,matdescra,Jij.ptrVal(),
        Jij.ptrCol(), Jij.ptrB(),Jij.ptrE(),s.ptr(),&zero,h.ptr());
#else
    jams_dcsrmv(transa,nspins3,nspins3,1.0,matdescra,Jij.ptrVal(),
        Jij.ptrCol(), Jij.ptrB(),Jij.ptrE(),s.ptr(),0.0,h.ptr());
#endif
  } else {
    std::fill(h.ptr(),h.ptr()+nspins3,0.0); 
  }

  // normalize by the gyroscopic factor
  for(i=0; i<nspins; ++i) {
    for(j=0; j<3;++j) {
      h(i,j) = ( h(i,j) + (w(i,j) + h_app[j])*mus(i))*gyro(i);
    }
  }
}
