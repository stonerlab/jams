
void jams_dcsrmv(const char trans[1], const int m, const int n, 
    const double alpha, const char descra[6], const double *val, 
    const int *indx, const int *ptrb, const int *ptre, double *x, 
    const double beta, double * y)
{
  // symmetric matrix
  if (descra[0] == 'S') {
    // upper matrix
    if(descra[1] == 'U') {
      for(int i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        for(int j=ptrb[i]; j<ptre[i]; ++j) {
          int k = indx[j];  // column
          if( i == k ){
            // diagonal
            y[i] = y[i] + alpha*x[k]*val[j];
          }
          else if ( i < k ){
            // upper triangle
            y[i] = y[i] + alpha*x[k]*val[j];
            y[k] = y[k] + alpha*x[i]*val[j]; 
          }
        }
      }
    }
    // lower matrix
    else if(descra[1] == 'L') {
      for(int i=0; i<m; ++i) { // iterate rows
        y[i] = beta * y[i];
        for(int j=ptrb[i]; j<ptre[i]; ++j) {
          int k = indx[j];  // column
          if( i == k ){
            // diagonal
            y[i] = y[i] + alpha*x[k]*val[j];
          }
          else if ( i > k ){
            // upper triangle
            y[i] = y[i] + alpha*x[k]*val[j];
            y[k] = y[k] + alpha*x[i]*val[j]; 
          }
        }
      }

    }
  // general matrix
  } else {
    for(int i=0; i<m; ++i) { // iterate rows
      y[i] = beta * y[i];
      for(int j=ptrb[i]; j<ptre[i]; ++j) {
        int k = indx[j];  // column
        y[i] = y[i] + alpha*x[k]*val[j];
      }
    }
  }

}
