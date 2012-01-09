#include "fields.h"
#include "globals.h"
#include "sparsematrix.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


// The factor of 2 is included in this function
// i.e. E = J_{2} sum_{ij}(\vec{s}_i \cdot \vec{s}_j)^2
//      \vec{H} = 2 J_{2} sum_{ij}(\vec{s}_i \cdot \vec{s}_j)
#ifdef CUDA
void calc_scalar_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre)
#else
void calc_scalar_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre)
#endif
{
  using namespace globals;

  double tmp;
  int i,j,k,l;
  int begin,end;

  if( J2ij.nonZero() > 0 ){

    for(i=0; i<nspins; ++i) { // iterate rows
      begin = ptrb[i]; end = ptre[i];
      for(j=begin; j<end; ++j) {
        k = indx[j];  // column
        // upper triangle and diagonal
        tmp = (s(i,0)*s(k,0) + s(i,1)*s(k,1) + s(i,2)*s(k,2))*val[j];
        if ( i > (k-1) ){
          for(l=0; l<3; ++l){
            h(i,l) = h(i,l) + s(k,l)*tmp;
          }
        }
      }
      for(j=begin; j<end; ++j) {
        k = indx[j];  // column
        // upper triangle and diagonal
        tmp = (s(i,0)*s(k,0) + s(i,1)*s(k,1) + s(i,2)*s(k,2))*val[j];
        if ( i > k ){
          for(l=0; l<3; ++l){
            h(k,l) = h(k,l) + s(i,l)*tmp;
          }
        }
      }
    }
  }
}

void calc_tensor_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre){

  // NOTE: Tensor calculations are added to the existing fields
  using namespace globals;
  assert(D2ii.nonZero() > 0);
 
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  char transa[1] = {'N'};
  char matdescra[6] = {'S','L','N','C','N','N'};
#ifdef MKL
  double one=1.0;
    mkl_dcsrmv(transa,&nspins3,&nspins3,&one,matdescra,Jij.valPtr(),
        D2ii.colPtr(), D2ii.ptrB(),D2ii.ptrE(),s.ptr(),&one,h.ptr());
#else
    jams_dcsrmv(transa,nspins3,nspins3,1.0,matdescra,Jij.valPtr(),
        D2ii.colPtr(), D2ii.ptrB(),D2ii.ptrE(),s.ptr(),1.0,h.ptr());
#endif
}

void calc_tensor_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre)
{
  // NOTE: this resets the field array to zero
  using namespace globals;
  assert(Jij.nonZero() > 0);
 
  char transa[1] = {'N'};
  char matdescra[6] = {'S','L','N','C','N','N'};
#ifdef MKL
  double one=1.0;
  double zero=0.0;
    mkl_dcsrmv(transa,&nspins3,&nspins3,&one,matdescra,Jij.valPtr(),
        Jij.colPtr(), Jij.ptrB(),Jij.ptrE(),s.ptr(),&zero,h.ptr());
#else
    jams_dcsrmv(transa,nspins3,nspins3,1.0,matdescra,Jij.valPtr(),
        Jij.colPtr(), Jij.ptrB(),Jij.ptrE(),s.ptr(),0.0,h.ptr());
#endif
}
void calculate_fields()
{
  using namespace globals;
  int i,j;

  // bilinear interactions
  if(Jij.nonZero() > 0) {
    // implicitly sets h = 0 in function above
    calc_tensor_bilinear(Jij.valPtr(),Jij.colPtr(),Jij.ptrB(),Jij.ptrE());
  } else {
    std::fill(h.ptr(),h.ptr()+nspins3,0.0); 
  }

  // fields added to h array
  calc_tensor_biquadratic(D2ii.valPtr(),D2ii.colPtr(),D2ii.ptrB(),D2ii.ptrE());
  calc_scalar_biquadratic(J2ij.valPtr(),J2ij.colPtr(),J2ij.ptrB(),J2ij.ptrE());

  // normalize by the gyroscopic factor
  for(i=0; i<nspins; ++i) {
    for(j=0; j<3;++j) {
      h(i,j) = ( h(i,j) + (w(i,j) + h_app[j])*mus(i))*gyro(i);
    }
  }
}
