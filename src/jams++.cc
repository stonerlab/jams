#define GLOBALORIGIN

#include <string>

#include "globals.h"
#include "utils.h"
#include "solver.h"
#include "lattice.h"

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

  rng.seed(time(NULL));

  lattice.createFromConfig();

  solver = Solver::Create();

  const double dt = (1E-16);
  solver->initialise(argc,argv,dt);

  return 0;
}

void jams_run() {
  using namespace globals;
  

  output.write("Running solver\n");

  double mag[3];
  for(int i=0; i<10000; ++i) {
    solver->run();
    if( (i%1000) == 0 ){
      mag[0] = 0.0; mag[1] = 0.0; mag[2] = 0.0;
      for(int n=0;n<nspins;++n) {
        for(int j=0; j<3; ++j) {
          mag[j] += s(n,j); 
        }
      }
    for(int j=0;j<3;++j) {
      mag[j] = mag[j]/static_cast<double>(nspins); 
    }
    double modmag = sqrt(mag[0]*mag[0]+mag[1]*mag[1]+mag[2]*mag[2]);
    output.write("%f %f %f %f \n",mag[0],mag[1],mag[2],modmag);
    }
  }
//  for(int i=0;i<nspins;++i) {
//    output.write("%f %f %f \n",s(i,0),s(i,1),s(i,2));
//  }
}

void jams_finish() {
  using namespace globals;

  if(solver != NULL) { delete solver; }
}

int main(int argc, char **argv) {

  jams_init(argc,argv);

  jams_run();

  jams_finish();
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

  jams_finish();
  exit(EXIT_FAILURE);
}
