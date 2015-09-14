// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/maths.h"
#include "core/utils.h"

void matrix_invert(const double in[3][3], double out[3][3]) {
  double det = in[0][0]*(in[1][1]*in[2][2]-in[1][2]*in[2][1])
             +in[0][1]*(in[1][2]*in[2][0]-in[1][0]*in[2][2])
             +in[0][2]*(in[1][0]*in[2][1]-in[1][1]*in[2][0]);

  det = 1.0/det;

  out[0][0] = det*(in[1][1]*in[2][2]-in[1][2]*in[2][1]);
  out[1][0] = det*(in[0][2]*in[1][2]-in[1][0]*in[2][2]);
  out[2][0] = det*(in[1][0]*in[2][1]-in[2][0]*in[1][1]);

  out[0][1] = det*(in[2][1]*in[0][2]-in[0][1]*in[2][2]);
  out[1][1] = det*(in[0][0]*in[2][2]-in[0][2]*in[2][0]);
  out[2][1] = det*(in[2][0]*in[0][1]-in[0][0]*in[2][1]);

  out[0][2] = det*(in[0][1]*in[1][2]-in[1][1]*in[0][2]);
  out[1][2] = det*(in[1][0]*in[0][2]-in[0][0]*in[1][2]);
  out[2][2] = det*(in[0][0]*in[1][1]-in[1][0]*in[0][1]);
}


// greatest common divisor
// https://stackoverflow.com/questions/4229870/c-algorithm-to-calculate-least-common-multiple-for-multiple-numbers
long gcd(long a, long b) {
  for (;;) {
    if (a == 0) {
      return b;
    }
    b %= a;
    if (b == 0) {
      return a;
    }
    a %= b;
  }
}

// lowest common multiple
// https://stackoverflow.com/questions/4229870/c-algorithm-to-calculate-least-common-multiple-for-multiple-numbers
long lcm(long a, long b) {
  long temp = gcd(a, b);

  if (temp == 0) {
    return 0;
  }

  return (a / temp * b);
}


double approximate_float_as_fraction(long &nominator, long &denominator, const double real, const long max_denominator) {
// https://www.ics.uci.edu/~eppstein/numth/frap.c
// find rational approximation to given real number
// David Eppstein / UC Irvine / 8 Aug 1993
// With corrections from Arno Formella, May 2008

// based on the theory of continued fractions
// if x = a1 + 1/(a2 + 1/(a3 + 1/(a4 + ...)))
// then best approximation is found by truncating this series
// (with some adjustments in the last term).
// Note the fraction can be recovered as the first column of the matrix
//  ( a1 1 ) ( a2 1 ) ( a3 1 ) ...
//  ( 1  0 ) ( 1  0 ) ( 1  0 )
// Instead of keeping the sequence of continued fraction terms,
// we just keep the last partial product of these matrices.
      long m[2][2];
      double x = real;
      long ai;

      /* initialize matrix */
      m[0][0] = m[1][1] = 1;
      m[0][1] = m[1][0] = 0;

      /* loop finding terms until denom gets too big */
      while (m[1][0] *  ( ai = (long)x ) + m[1][1] <= max_denominator) {
    long t;
    t = m[0][0] * ai + m[0][1];
    m[0][1] = m[0][0];
    m[0][0] = t;
    t = m[1][0] * ai + m[1][1];
    m[1][1] = m[1][0];
    m[1][0] = t;
          if(x==(double)ai) break;     // AF: division by zero
    x = 1/(x - (double) ai);
          if(x>(double)0x7FFFFFFF) break;  // AF: representation failure
      }

      /* now remaining x is between 0 and 1/ai */
      /* approx as either 0 or 1/m where m is max that will fit in max_denominator */
      /* first try zero */
      nominator = m[0][0];
      denominator = m[1][0];
      return real - ((double) m[0][0] / (double) m[1][0]);
}



double legendre_poly(const double x, const int n) {

  switch (n)
  {
    case 0:
      return legendre_poly_0(x);
    case 1:
      return legendre_poly_1(x);
    case 2:
      return legendre_poly_2(x);
    case 3:
      return legendre_poly_3(x);
    case 4:
      return legendre_poly_4(x);
    case 5:
      return legendre_poly_5(x);
    case 6:
      return legendre_poly_6(x);
  }

  if (unlikely(x == 1.0)) {
    return 1.0;
  }

  if (unlikely(x == -1.0)) {
    if (n % 2) {
      return -1.0;
    } else {
      return 1.0;
    }
  }

  if (unlikely(x == 0.0) && n % 2) {
    return 0.0;
  }

  // http://www.storage-b.com/math-numerical-analysis/18
  double pn1(legendre_poly_2(x));
  double pn2(legendre_poly_1(x));
  double pn(pn1);

  for (int l = 3; l < n + 1; ++l) {
    pn =  (((2.0 * l) - 1.0) * x * pn1 - ((l - 1.0) * pn2)) / static_cast<double>(l);
    pn2 = pn1;
    pn1 = pn;
  }

  return pn;
}


double legendre_dpoly(const double x, const int n) {

  switch (n)
  {
    case 0:
      return legendre_dpoly_0(x);
    case 1:
      return legendre_dpoly_1(x);
    case 2:
      return legendre_dpoly_2(x);
    case 3:
      return legendre_dpoly_3(x);
    case 4:
      return legendre_dpoly_4(x);
    case 5:
      return legendre_dpoly_5(x);
    case 6:
      return legendre_dpoly_6(x);
  }

  if (unlikely(x == 0.0) && n % 2 == 0) {
    return 0.0;
  }

  return (x * legendre_poly(x, n) - legendre_poly(x, n - 1)) / static_cast<double>(2 * n + 1);
}
