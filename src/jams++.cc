#define GLOBALORIGIN

#include <string>

#include "globals.h"
#include "utils.h"

std::string seedname;

int jams_init(int argc, char **argv)
{
  if(argc == 1) {
    // seedname is executable
    seedname = std::string(argv[0]);
  } else {
    // seedname is first argument
    seedname = std::string(argv[1]);
  }
  trim(seedname);

  output.open("%s.out",seedname.c_str());

  output.write("\nJAMS++\n");
  output.write("Compiled %s, %s\n",__DATE__,__TIME__);

  return 0;
}

int main(int argc, char **argv)
{
  jams_init(argc,argv);

  return 0;
}
