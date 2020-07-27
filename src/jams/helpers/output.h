//
// Created by Joe Barker on 2018/08/28.
//

#ifndef JAMS_HELPERS_OUTPUT_H
#define JAMS_HELPERS_OUTPUT_H
#include <ostream>
#include <iomanip>

namespace jams {
    namespace output {
        void initialise();

        std::string full_path_filename(const std::string& ending);
        std::string full_path_filename_series(const std::string& ending, int num, int width=7);
        std::string output_path();
    }

    namespace fmt {
        inline std::ostream &integer(std::ostream &os) {
          return os << std::fixed << std::setw(8) << std::right;
        }

        inline std::ostream &fixed_integer(std::ostream &os) {
          return os << std::fixed << std::setw(8);
        }

        inline std::ostream &decimal(std::ostream &os) {
          return os << std::setprecision(6) << std::setw(16) << std::fixed << std::right;
        }

        inline std::ostream &sci(std::ostream &os) {
          return os << std::setprecision(8) << std::setw(16) << std::scientific << std::right;
        };
    }
}

#endif //JAMS_HELPERS_OUTPUT_H
