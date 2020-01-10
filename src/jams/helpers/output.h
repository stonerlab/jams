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
    }

    namespace fmt {
        inline std::ostream &integer(std::ostream &os) {
          return os << std::fixed;
        }

        inline std::ostream &fixed_integer(std::ostream &os) {
          return os << std::fixed << std::setw(8);
        }

        inline std::ostream &decimal(std::ostream &os) {
          return os << std::setprecision(6) << std::setw(12) << std::fixed;
        }

        inline std::ostream &sci(std::ostream &os) {
          return os << std::setprecision(8) << std::setw(12) << std::scientific;
        };
    }
}

#endif //JAMS_HELPERS_OUTPUT_H
