//
// Created by Joe Barker on 2017/11/20.
//

#include "unitcell.h"

double volume(const UnitCell &u) {
  return scalar_triple_product(u.a(), u.b(), u.c());
}