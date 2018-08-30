//
// Created by Joe Barker on 2018/08/28.
//

#include <iostream>

#include "output.h"

namespace jams {
    void desync_io() {
      std::cin.tie(nullptr);
      std::ios_base::sync_with_stdio(false);
    }
}

