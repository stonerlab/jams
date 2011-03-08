#define GLOBALORIGIN

#include <string>
#include <cstdarg>

#include "solver.h"
#include "physics.h"
#include "globals.h"
#include "utils.h"
#include "lattice.h"
#include "monitor.h"
#include "boltzmann.h"
#include "boltzmann_mag.h"
#include "magnetisation.h"

#ifdef CUDA
#include <cublas.h>
#endif

namespace {
  Solver *solver;
  Physics *physics;
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

#ifdef DEBUG
  output.write("DEBUG Build\n");
#endif

  std::string cfgfile = seedname+".cfg";

  {
    libconfig::Config config;

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
      jams_error("Undefined config error");
    }

    libconfig::Setting &phys = config.lookup("physics");


    std::string solname;
    std::string physname;
    unsigned int randomseed;

    double init_temperature=0.0;

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

      globals::h_app[0] = config.lookup("sim.h_app.[0]");
      globals::h_app[1] = config.lookup("sim.h_app.[1]");
      globals::h_app[2] = config.lookup("sim.h_app.[2]");
    
      if( config.exists("sim.solver") == true ) {
        config.lookupValue("sim.solver",solname);
        std::transform(solname.begin(),solname.end(),solname.begin(),toupper);
      }

      if( config.exists("sim.physics") == true ) {
        config.lookupValue("sim.physics",physname);
        std::transform(physname.begin(),physname.end(),physname.begin(),toupper);
      }


      if( config.exists("sim.seed") == true) {
        config.lookupValue("sim.seed",randomseed);
        output.write("Random generator seeded from config file\n");
      } else {
        randomseed = time(NULL);
        output.write("Random generator seeded from time\n");
      }
      output.write("Seed: %d\n",randomseed);

      init_temperature = config.lookup("sim.temperature");
      output.write("Initial temperature: %f\n",init_temperature);

    }
    catch(const libconfig::SettingNotFoundException &nfex) {
      jams_error("Setting '%s' not found",nfex.getPath());
    }
    catch(...) {
      jams_error("Undefined config error");
    }

    rng.seed(randomseed);

    lattice.createFromConfig(config);
  
    if(physname == "FMR") {
      physics = Physics::Create(FMR);
    } else {
      physics = Physics::Create(EMPTY);
      output.write("WARNING: Using empty physics package\n");
    }

    
    physics->init(phys);

      
    if(solname == "HEUNLLG") {
      solver = Solver::Create(HEUNLLG);
    }
    else if (solname == "HEUNLLMS") {
      solver = Solver::Create(HEUNLLMS);
    }
    else if (solname == "SEMILLG") {
      solver = Solver::Create(SEMILLG);
    }
    else if (solname == "CUDASEMILLG") {
      solver = Solver::Create(CUDASEMILLG);
    }
    else if (solname == "FFTNOISE") {
      solver = Solver::Create(FFTNOISE);
    }
    else { // default
      output.write("WARNING: Using default solver (HEUNLLG)\n");
      solver = Solver::Create();
    }

    solver->initialise(argc,argv,dt);
    solver->setTemperature(init_temperature);

  }
  return 0;
}

void jams_run() {
  using namespace globals;
  

  h_app[0] = 0.0; h_app[1] = 0.0; h_app[2] = 0.0;

  Monitor *mag = new MagnetisationMonitor();
  mag->initialise();

  output.write("\n----Equilibration----\n");
  output.write("Running solver\n");
  for(unsigned int i=0;i<steps_eq;++i) {
    if( ((i)%steps_out) == 0 ){
      solver->syncOutput();
      mag->write(solver->getTime());
    }
    physics->run(solver->getTime(),dt);
    solver->run();
  }
  
  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  std::clock_t start = std::clock();
  for(unsigned int i=0; i<steps_run; ++i) {
    if( ((i)%steps_out) == 0 ){
      solver->syncOutput();
      mag->write(solver->getTime());
      physics->monitor(solver->getTime(),dt);
    }
    physics->run(solver->getTime(),dt);
    solver->run();
  }
  double elapsed = static_cast<double>(std::clock()-start);
  elapsed/=CLOCKS_PER_SEC;
  output.write("Solving time: %f\n",elapsed);

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
