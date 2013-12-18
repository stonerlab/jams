#ifndef SYS_ARGUMENTS_H
#define SYS_ARGUMENTS_H

#include <string>
#include <iostream>
#include <cstdlib>

#include "sys_defines.h"
#include "sys_types.h"

// see stack overflow "Parse Command Line Argumetns [duplicate]"

namespace jbLib {
  class jbArguments{
    public:
      static JB_INLINE void setArgumentString(int argc, char * argv[]){
        for(int32 i=1; i<argc; ++i){
          argumentString += argv[i];
          argumentString += " ";
        }
      }

      static JB_INLINE const std::string& getArgumentString(){
        return argumentString;
      }

      static JB_INLINE bool flagExists(const std::string& flag){
        return (argumentString.find(flag) != std::string::npos);

      }
      static JB_INLINE std::string getFlagValue(const std::string& flag){
        size_t flagPos=0;
        size_t equalPos=0;
        size_t nextFlagPos=0;

        if( flagExists(flag) ){
          flagPos = argumentString.find(flag);
          equalPos = argumentString.find("=",flagPos);
          nextFlagPos = argumentString.find(" -",flagPos);

          std::cout<<flagPos<<"\t"<<nextFlagPos<<std::endl;

          if( equalPos != nextFlagPos ){
            return (argumentString.substr(equalPos+1,nextFlagPos - equalPos));
          }
        }
        // return blank if no option is found or option was left blank on
        // the command line (i.e. -x= )
        return "";
      }

    private:

      static std::string argumentString;

  };
}
#endif
