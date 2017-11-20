//
// Created by Joe Barker on 2017/11/20.
//

#include "unitcell.h"

template <class Type>
double volume(const UnitCell<Type> &u) {
  return scalar_triple_product(u.a(), u.b(), u.c());
}