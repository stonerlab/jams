//
// Created by Joe Barker on 2018/08/28.
//

#include <iostream>

#include "output.h"

namespace jams {
    namespace output {
        void desync_io() {
          std::cin.tie(nullptr);
          std::ios_base::sync_with_stdio(false);
        }

        void set_default_cout_flags() {
          std::cout << std::boolalpha;
        }

        void initialise() {
          desync_io();
          set_default_cout_flags();
        }
    }

}

