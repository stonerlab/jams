// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/maths.h"

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
