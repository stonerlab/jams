// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_SYS_ARGUMENTS_H
#define JBLIB_SYS_ARGUMENTS_H

#include <string>
#include <iostream>
#include <cstdlib>

#include "jblib/sys/types.h"


// see stack overflow "Parse Command Line Argumetns [duplicate]"

namespace jblib {
  class jbArguments{
   public:
    static inline void setArgumentString(int argc, char * argv[]) {
      for (int32 i = 1; i < argc; ++i) {
        argumentString += argv[i];
        argumentString += " ";
      }
    }

    static inline const std::string& getArgumentString() {
      return argumentString;
    }

    static inline bool flagExists(const std::string& flag) {
      return (argumentString.find(flag) != std::string::npos);
    }

    static inline std::string getFlagValue(const std::string& flag) {
      if (flagExists(flag)) {
        size_t flagPos = 0, equalPos = 0, nextFlagPos = 0;
        flagPos = argumentString.find(flag);
        equalPos = argumentString.find("=", flagPos);
        nextFlagPos = argumentString.find(" -", flagPos);

        std::cout << flagPos << "\t" << nextFlagPos << std::endl;

        if (equalPos != nextFlagPos) {
          return (argumentString.substr(equalPos+1, nextFlagPos - equalPos));
        }
      }
        // return blank if no option is found or option was left blank on
        // the command line (i.e. -x= )
      return "";
    }

   private:
    static std::string argumentString;
  };
}  // namespace jblib
#endif  // JBLIB_SYS_ARGUMENTS_H
