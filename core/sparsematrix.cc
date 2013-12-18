#include "globals.h"

void jams_dcsrmv(const char trans[1], const int m, const int n, 
    const double alpha, const char descra[6], const double *val, 
    const int *indx, const int *ptrb, const int *ptre, double *x, 
    const double beta, double * y)
{
  // symmetric matrix
  int i,j,k;
  int begin,end;
  double tmp;
  if (descra[0] == 'S') {
    // upper matrix
    if(descra[1] == 'L') {
      for(i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i > (k-1) ){
            y[i] = y[i] + x[k]*tmp;
          }
        }
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i > k ){
            y[k] = y[k] + x[i]*tmp; 
          }
        }
      }
    }
    // lower matrix
    else if(descra[1] == 'U') {
      output.write("WARNING: dcsrmv with 'S' and 'U' is untested.\n");
      for(i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if ( i < (k+1) ){
            y[i] = y[i] + x[k]*tmp;;
          }
        }
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i < k ){
            y[k] = y[k] + x[i]*tmp; 
          }
        }
      }
    }
  // general matrix
  } else {
    output.write("WARNING: general dcsrmv is untested.\n");
    for(i=0; i<m; ++i) { // iterate rows
      y[i] = beta * y[i];
      begin = ptrb[i]; end = ptre[i];
      for(j=begin; j<end; ++j) {
        k = indx[j];  // column
        y[i] = y[i] + alpha*x[k]*val[j];
      }
    }
  }

}

// TEMPORARY HACK FOR CUDA COMPATABILITY
void jams_dcsrmv(const char trans[1], const int m, const int n, 
    const double alpha, const char descra[6], const float *val, 
    const int *indx, const int *ptrb, const int *ptre, double *x, 
    const double beta, double * y)
{
  // symmetric matrix
  int i,j,k;
  int begin,end;
  double tmp;
  if (descra[0] == 'S') {
    // upper matrix
    if(descra[1] == 'L') {
      for(i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i > (k-1) ){
            y[i] = y[i] + x[k]*tmp;
          }
        }
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i > k ){
            y[k] = y[k] + x[i]*tmp; 
          }
        }
      }
    }
    // lower matrix
    else if(descra[1] == 'U') {
      output.write("WARNING: dcsrmv with 'S' and 'U' is untested.\n");
      for(i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        begin = ptrb[i]; end = ptre[i];
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // lower triangle and diagonal
          tmp = alpha*val[j];
          if ( i < (k+1) ){
            y[i] = y[i] + x[k]*tmp;;
          }
        }
        for(j=begin; j<end; ++j) {
          k = indx[j];  // column
          // upper triangle and diagonal
          tmp = alpha*val[j];
          if ( i < k ){
            y[k] = y[k] + x[i]*tmp; 
          }
        }
      }
    }
  // general matrix
  } else {
    output.write("WARNING: general dcsrmv is untested.\n");
    for(i=0; i<m; ++i) { // iterate rows
      y[i] = beta * y[i];
      begin = ptrb[i]; end = ptre[i];
      for(j=begin; j<end; ++j) {
        k = indx[j];  // column
        y[i] = y[i] + alpha*x[k]*val[j];
      }
    }
  }

}
