//
// Created by Joe Barker on 2017/11/30.
//

#include <pcg_random.hpp>
#include <jams/interface/randutils.h>
#include <jams/helpers/random.h>

pcg32 &jams::random_generator() {
  static pcg32 rng{randutils::auto_seed_128{}.base()};
  return rng;
}
