//
// Created by Joe Barker on 2017/11/30.
//

#include <pcg/pcg_random.hpp>
#include <randutils.h>
#include <jams/helpers/random.h>

pcg32 &jams::random_generator() {
  static pcg32 rng{randutils::auto_seed_128{}.base()};
  return rng;
}
