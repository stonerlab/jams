#define GLOBALORIGIN

#include <string>
#include <cstdarg>

#include "solver.h"
#include "globals.h"
#include "utils.h"
#include "lattice.h"
#include "monitor.h"
#include "boltzmann.h"
#include "boltzmann_mag.h"

std::string seedname;

namespace {
  Solver *solver;
  double dt=0.0;
  unsigned int steps_eq=0;
  unsigned int steps_run=0;
  unsigned int steps_out=0;
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

  std::string solname;

  try {
    dt = config.lookup("sim.t_step");
    output.write("Timestep: %e\n",dt);

    double tmp = config.lookup("sim.t_eq");
    steps_eq = static_cast<unsigned int>(tmp/dt);
    output.write("Equilibration time: %e (%d steps)\n",tmp,steps_eq);

    tmp = config.lookup("sim.t_run");
    steps_run = static_cast<unsigned int>(tmp/dt);
    output.write("Run time: %e (%d steps)\n",tmp,steps_run);
    
    tmp = config.lookup("sim.t_out");
    steps_out = static_cast<unsigned int>(tmp/dt);
    output.write("Output time: %e (%d steps)\n",tmp,steps_out);
  
    if( config.exists("sim.solver") == true ) {
      config.lookupValue("sim.solver",solname);
      std::transform(solname.begin(),solname.end(),solname.begin(),toupper);
    }

  }
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }
  catch(...) {
    jams_error("Undefined error");
  }


  
    
  if(solname == "HEUNLLG") {
    solver = Solver::Create(HEUNLLG);
  }
  else if (solname == "HEUNLLMS") {
    solver = Solver::Create(HEUNLLMS);
  }
  else if (solname == "SEMILLG") {
    solver = Solver::Create(SEMILLG);
  }
  else if (solname == "FFTNOISE") {
    solver = Solver::Create(FFTNOISE);
  }
  else { // default
    output.write("WARNING: Using default solver (HEUNLLG)\n");
    solver = Solver::Create();
  }
  
  solver->initialise(argc,argv,dt);


  return 0;
}

void jams_run() {
  using namespace globals;
  

  h_app[0] = 0.0; h_app[1] = 0.0; h_app[2] = 0.1*boltzmann_si/mus(0);

  output.write("\n----Equilibration----\n");
  output.write("Running solver\n");
  for(int i=0;i<steps_eq;++i) {
    solver->run();
  }
  
  Monitor *mon = new BoltzmannMagMonitor();
  mon->initialise();

  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  double mag[3];
  for(int i=0; i<steps_run; ++i) {
    if( ((i+1)%steps_out) == 0 ){
      mon->write();
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
      output.write("%f %f %f %1.12f \n",mag[0],mag[1],mag[2],modmag);
    }
    solver->run();
    mon->run();
  }

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
