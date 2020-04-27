//
// Created by Joseph Barker on 2020-04-27.
//

#ifndef JAMS_TEST_OUTPUT_H
#define JAMS_TEST_OUTPUT_H

#include <sstream>
#include <iostream>

namespace jams {
    namespace testing {
        inline void toggle_cout() {
          static bool cout_is_suppressed = false;
          static std::stringstream null_buffer;
          static std::streambuf *sbuf;

          if (!cout_is_suppressed) {
            sbuf = std::cout.rdbuf();
            std::cout.rdbuf(null_buffer.rdbuf());
            cout_is_suppressed = true;
          } else {
            std::cout.rdbuf(sbuf);
            null_buffer.str("");
            cout_is_suppressed = false;
          }
        }
    }
}
#endif //JAMS_TEST_OUTPUT_H
