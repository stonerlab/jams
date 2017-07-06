//
// Created by Joe Barker on 2017/07/05.
//


#include <stdexcept>
#include <jams/core/maths.h>

namespace {
    //http://easyspin.org/documentation/stevensoperators.html
  // O_k^q(J)
    double stevens_operator_2_0(double J, double Jx, double Jy, double Jz) {
      return 3 * pow2(Jz) - J * (J + 1);
    }

    double stevens_operator_4_0(double J, double Jx, double Jy, double Jz) {
      double X = J * (J + 1);
      return 35 * pow4(Jz)
             - (30 * X - 25) * pow2(Jz)
             + 3 * pow2(X)
             - 6 * X;
    }

    double stevens_operator_6_0(double J, double Jx, double Jy, double Jz) {
      double X = J * (J + 1);
      return 231 * pow6(Jz)
             - (315 * X - 735) * pow4(Jz)
             + (105 * pow2(X) - 525 * X + 294) * pow2(Jz)
              - 5 * pow3(X)
              + 40 * pow2(X)
              - 60 * X;
    }
}

double stevens_operator(double J, double Jx, double Jy, double Jz, int q, int k) {

  // process k == 0 first as these are most common for magnetism (uniaxial)
  if (k == 0) {
    if (q == 2) {
      return stevens_operator_2_0(J, Jx, Jy, Jz);
    }

    if (q == 4) {
      return stevens_operator_4_0(J, Jx, Jy, Jz);
    }

    if (q == 6) {
      return stevens_operator_6_0(J, Jx, Jy, Jz);
    }
  }

  // k == p are next most common for magnetism (cubic)
  if (k == q) {
    if (q == 2) {
      return stevens_operator_2_2(J, Jx, Jy, Jz);
    }
  }

  throw std::invalid_argument("");
}
