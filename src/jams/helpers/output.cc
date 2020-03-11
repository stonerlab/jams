//
// Created by Joe Barker on 2018/08/28.
//

#include <iostream>
#include <fstream>

#include "jams/common.h"
#include "jams/helpers/output.h"

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

        std::ofstream open_file(const std::string &filename, std::ios_base::openmode mode) {
          return std::ofstream(jams::instance().output_path() + "/" + filename, mode);
        }

        std::string output_path() {
          return jams::instance().output_path() + "/";
        }
    }

}

