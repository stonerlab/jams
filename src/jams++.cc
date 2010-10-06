#define GLOBALORIGIN

#include <string>

#include "globals.h"
#include "utils.h"
#include "solver.h"
#include "geometry.h"
#include "vecfield.h"

std::string seedname;

namespace {
  Solver *solver;
} // anon namespace

int jams_init(int argc, char **argv) {
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

  std::string cfgfile = seedname+".cfg";

  try {
    config.readFile(cfgfile.c_str());
  }
  catch(const libconfig::FileIOException &fioex) {
    jams_error("I/O error while reading '%s'", cfgfile.c_str());
  }
  catch(const libconfig::ParseException &pex) {
    jams_error("Error parsing %s:%i: %s", pex.getFile(), 
        pex.getLine(), pex.getError());
  }
  catch(...) {
    jams_error("Undefined error");
  }

  geometry.readFromConfig();


  solver = Solver::Create();

  const double dt = 0.01;
  solver->initialise(argc,argv,dt);
  return 0;
}

void jams_finish() {
  delete solver;
}

int main(int argc, char **argv) {
  jams_init(argc,argv);

  return EXIT_SUCCESS;
}

void jams_error(const char *string, ...) {

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
