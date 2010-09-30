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

  config.open("%s.cfg",seedname.c_str());

  return 0;
}

int main(int argc, char **argv)
{
  jams_init(argc,argv);

  return EXIT_SUCCESS;
}

void jams_error(const char *string, ...)
{

  // TODO: Fix this so that the arguments are passed through.
  va_list args;
  char buffer[1024];

  va_start(args,string);
    vsprintf(buffer, string, args);
  va_end(args);

  output.write("\n********** JAMS ERROR **********\n");
  output.write("%s\n",buffer);
  exit(EXIT_FAILURE);
}
