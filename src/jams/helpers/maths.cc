// Copyright 2014 Joseph Barker. All rights reserved.

#include "maths.h"
#include "utils.h"

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

    double p_lm2 = 1.0;
    double p_lm1 = x;
    double p_l = 0.0;

    for (unsigned int nn = 2; nn <= n; ++nn)
      {
        //  This arrangement is supposed to be better for roundoff
        //  protection, Arfken, 2nd Ed, Eq 12.17a.
        p_l = 2.0 * x * p_lm1 - p_lm2
              - (x * p_lm1 - p_lm2) / double(nn);
        p_lm2 = p_lm1;
        p_lm1 = p_l;
      }

      return p_l;
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

