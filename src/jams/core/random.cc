//
// Created by Joe Barker on 2017/11/30.
//

#include <pcg/pcg_random.hpp>
#include <jams/helpers/random.h>

pcg32 &jams::random_generator() {
  static pcg32 rng = pcg_extras::seed_seq_from<std::random_device>();
  return rng;
}
