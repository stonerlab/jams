#include "fields.h"
#include "globals.h"
#include "sparsematrix.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif

#ifdef CUDA
void calc_scalar_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#else
void calc_scalar_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#endif
{
  using namespace globals;

  int i,j,k,l;
  int begin,end;


  for(i=0; i<nspins; ++i) { // iterate rows
    begin = ptrb[i]; end = ptre[i];
    for(j=begin; j<end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      if ( i > (k-1) ){
        for(l=0; l<3; ++l){
          y(i,l) = y(i,l) + s(k,l)*val[j];
        }
      }
    }
    for(j=begin; j<end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      if ( i > k ){
        for(l=0; l<3; ++l){
          y(k,l) = y(k,l) + s(i,l)*val[j];
        }
      }
    }
  }
}

#ifdef CUDA
void calc_scalar_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#else
void calc_scalar_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#endif
{
  // NOTE: Factor of two is included here for biquadratic terms
  using namespace globals;

  double tmp;
  int i,j,k,l;
  int begin,end;


  for(i=0; i<nspins; ++i) { // iterate rows
    begin = ptrb[i]; end = ptre[i];
    for(j=begin; j<end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      tmp = (s(i,0)*s(k,0) + s(i,1)*s(k,1) + s(i,2)*s(k,2))*val[j];
      if ( i > (k-1) ){
        for(l=0; l<3; ++l){
          y(i,l) = y(i,l) + 2.0*s(k,l)*tmp;
        }
      }
    }
    for(j=begin; j<end; ++j) {
      k = indx[j];  // column
      // upper triangle and diagonal
      tmp = (s(i,0)*s(k,0) + s(i,1)*s(k,1) + s(i,2)*s(k,2))*val[j];
      if ( i > k ){
        for(l=0; l<3; ++l){
          y(k,l) = y(k,l) + 2.0*s(i,l)*tmp;
        }
      }
    }
  }
}

#ifdef CUDA
void calc_tensor_biquadratic(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#else
void calc_tensor_biquadratic(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#endif
{
  // NOTE: Factor of two is included here for biquadratic terms
  // NOTE: Tensor calculations are added to the existing fields
  using namespace globals;
 
  // dscrmv below has beta=0.0 -> field array is zeroed
  // exchange
  char transa[1] = {'N'};
  char matdescra[6] = {'S','L','N','C','N','N'};
#ifdef MKL
  double one=1.0;
  double two=2.0;
    mkl_dcsrmv(transa,&nspins3,&nspins3,&two,matdescra,val,
        indx, ptrb,ptre,s.data(),&zero,y.data());
#else
    jams_dcsrmv(transa,nspins3,nspins3,2.0,matdescra,val,
        indx, ptrb,ptre,s.data(),1.0,y.data());
#endif
}


#ifdef CUDA
void calc_tensor_bilinear(const float *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#else
void calc_tensor_bilinear(const double *val, const int *indx, 
  const int *ptrb, const int *ptre, jbLib::Array<double,2> &y)
#endif
{
  // NOTE: this resets the field array to zero
  using namespace globals;
 
  char transa[1] = {'N'};
  char matdescra[6] = {'S','L','N','C','N','N'};
#ifdef MKL
  double one=1.0;
  double one=1.0;
    mkl_dcsrmv(transa,&nspins3,&nspins3,&one,matdescra,val,
        indx, ptrb,ptre,s.data(),&zero,y.data());
#else
    jams_dcsrmv(transa,nspins3,nspins3,1.0,matdescra,val,
        indx, ptrb,ptre,s.data(),1.0,y.data());
#endif
}
void calculate_fields()
{
  using namespace globals;
  int i,j;

  std::fill(h.data(),h.data()+nspins3,0.0); 
  if(J1ij_s.nonZero() > 0) {
    calc_scalar_bilinear(J1ij_s.valPtr(),J1ij_s.colPtr(),J1ij_s.ptrB(),J1ij_s.ptrE(),h);
  } 
  if(J1ij_t.nonZero() > 0) {
    calc_tensor_bilinear(J1ij_t.valPtr(),J1ij_t.colPtr(),J1ij_t.ptrB(),J1ij_t.ptrE(),h);
  }
  if(J2ij_s.nonZero() > 0) {
    calc_scalar_biquadratic(J2ij_s.valPtr(),J2ij_s.colPtr(),J2ij_s.ptrB(),J2ij_s.ptrE(),h);
  }
  if(J2ij_t.nonZero() > 0) {
    calc_tensor_biquadratic(J2ij_t.valPtr(),J2ij_t.colPtr(),J2ij_t.ptrB(),J2ij_t.ptrE(),h);
  }

  // normalize by the gyroscopic factor
  for(i=0; i<nspins; ++i) {
    for(j=0; j<3;++j) {
      h(i,j) = ( h(i,j) + (w(i,j) + h_app[j])*mus(i))*gyro(i);
    }
  }
}
