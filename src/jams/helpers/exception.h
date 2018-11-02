#ifndef JAMS_CORE_EXCEPTION_H
#define JAMS_CORE_EXCEPTION_H

#include <stdexcept>
#include <sstream>
#include <cstring>

namespace jams {

  class runtime_error : std::runtime_error {
      std::string msg;
  public:
      runtime_error(const std::string &arg, const char *file, int line, const char *function="") :
      std::runtime_error(arg) {
          std::ostringstream o;
          if (strcmp(function, "") == 0) {
              o << file << ":" << line << ": " << arg ;
          } else {
              o << file << ":" << line << "[" << function << "]" << ": " << arg ;
          }
          msg = o.str();
      }
      ~runtime_error() throw() {}
      const char *what() const throw() {
          return msg.c_str();
      }
  };

  class unimplemented_error : public std::logic_error
  {
  public:
      explicit unimplemented_error(const std::string &func) : std::logic_error("unimplemented function: " + func) { };
  };
#define JAMS_UNIMPLEMENTED_FUNCTION throw jams::unimplemented_error(__PRETTY_FUNCTION__)
}

#endif  // JAMS_CORE_EXCEPTION_H
