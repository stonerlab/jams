#define GLOBALORIGIN
#define JAMS_VERSION "0.6.0"
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#include <string>
#include <cstdarg>
#include <iostream>
#include <fstream>

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
  unsigned long steps_eq=0;
  unsigned long steps_run=0;
  unsigned long steps_out=0;

  bool toggleVisualise=false;
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
  output.write("Version %s\n", JAMS_VERSION);
  output.write("Commit %s\n", QUOTEME(GITCOMMIT));
  output.write("Compiled %s, %s\n",__DATE__,__TIME__);
  output.write("%s\n", QUOTEME(GITCOMMIT));
  output.write("----------------------------------------\n");
  
  time_t rawtime;
  struct tm * timeinfo;
  char timebuffer[80];
  time( &rawtime );
  timeinfo = localtime( &rawtime );
  strftime(timebuffer,80,"%b %d %Y, %X",timeinfo);
  output.write("Run time %s\n",timebuffer);
  output.write("----------------------------------------\n");

#ifdef DEBUG
  output.write("\nDEBUG Build\n");
#endif

  output.write("\nReading configuration file...\n");

  std::string cfgfile = seedname+".cfg";

  output.write("  * Config file: %s\n",cfgfile.c_str());

  {
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



    std::string solname;
    std::string physname;
    unsigned int randomseed;

    double init_temperature=0.0;

    try {


      dt = config.lookup("sim.t_step");
      output.write("  * Timestep:           %1.8e\n",dt);


      double tmp = config.lookup("sim.t_eq");
      steps_eq = static_cast<unsigned long>(tmp/dt);
      output.write("  * Equilibration time: %1.8e (%lu steps)\n",tmp,steps_eq);

      tmp = config.lookup("sim.t_run");
      steps_run = static_cast<unsigned long>(tmp/dt);
      output.write("  * Run time:           %1.8e (%lu steps)\n",tmp,steps_run);
      
      tmp = config.lookup("sim.t_out");
      steps_out = static_cast<unsigned long>(tmp/dt);
      output.write("  * Output time:        %1.8e (%lu steps)\n",tmp,steps_out);

      globals::h_app[0] = config.lookup("sim.h_app.[0]");
      globals::h_app[1] = config.lookup("sim.h_app.[1]");
      globals::h_app[2] = config.lookup("sim.h_app.[2]");
    

      if( config.exists("sim.visualise") == true) {
        config.lookupValue("sim.visualise",toggleVisualise);
        output.write("  * Visualisation is ON\n");
      } else {
        toggleVisualise = false;
      }


      if( config.exists("sim.seed") == true) {
        config.lookupValue("sim.seed",randomseed);
        output.write("  * Random generator seeded from config file\n");
      } else {
        randomseed = time(NULL);
        output.write("  * Random generator seeded from time\n");
      }
      output.write("  * Seed: %d\n",randomseed);

      init_temperature = config.lookup("sim.temperature");
      globals::globalTemperature = init_temperature;
      output.write("  * Initial temperature: %f\n",init_temperature);


      rng.seed(randomseed);

      lattice.createFromConfig(config);

      if( config.exists("sim.solver") == true ) {
        config.lookupValue("sim.solver",solname);
        std::transform(solname.begin(),solname.end(),solname.begin(),toupper);
      }

      output.write("\nInitialising physics module...\n");
      if( config.exists("sim.physics") == true ) {
        config.lookupValue("sim.physics",physname);
        std::transform(physname.begin(),physname.end(),physname.begin(),toupper);

        if(physname == "FMR") {
          physics = Physics::Create(FMR);
        }else if(physname == "TTM") {
          physics = Physics::Create(TTM);
        }else if(physname == "SPINWAVES") {
          physics = Physics::Create(SPINWAVES);
        }else if(physname == "DYNAMICSF") {
          physics = Physics::Create(DYNAMICSF);
        }else if(physname == "SQUARE") {
          physics = Physics::Create(SQUARE);
        }else if(physname == "FIELDCOOL") {
          physics = Physics::Create(FIELDCOOL);
        }else{
          jams_error("Unknown Physics package selected.");
        }

        libconfig::Setting &phys = config.lookup("physics");
        physics->init(phys);

      } else {
        physics = Physics::Create(EMPTY);
        output.write("\nWARNING: Using empty physics package\n");
      }

    }
    catch(const libconfig::SettingNotFoundException &nfex) {
      jams_error("Setting '%s' not found",nfex.getPath());
    }
    catch(...) {
      jams_error("Undefined config error");
    }

    output.write("\nInitialising solver...\n");
    if(solname == "HEUNLLG") {
      solver = Solver::Create(HEUNLLG);
    }
    else if (solname == "CUDAHEUNLLG") {
      solver = Solver::Create(CUDAHEUNLLG);
    }
    else if (solname == "METROPOLISMC") {
        solver = Solver::Create(METROPOLISMC);
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
  
  Monitor *mag = new MagnetisationMonitor();
  mag->initialise();

  std::string name = "_eng.dat";
  name = seedname+name;
  std::ofstream engfile(name.c_str());

  double e1_s, e1_t, e2_s, e2_t;

  output.write("\n----Equilibration----\n");
  output.write("Running solver\n");
  for(unsigned long i=0;i<steps_eq;++i) {
    if( ((i)%steps_out) == 0 ){
      solver->syncOutput();
      mag->write(solver->getTime());
      solver->calcEnergy(e1_s,e1_t,e2_s,e2_t);
      engfile << solver->getTime() << "\t" << e1_s << "\t" << e1_t << "\t" << e2_s << "\t" << e2_t << std::endl;
    }
    physics->run(solver->getTime(),dt);
    solver->setTemperature(globalTemperature);
    solver->run();
  }
  
  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  std::clock_t start = std::clock();
  // int outcount = 0;
  for(unsigned long i=0; i<steps_run; ++i) {
    if( ((i)%steps_out) == 0 ){
      solver->syncOutput();
      mag->write(solver->getTime());
      physics->monitor(solver->getTime(),dt);
      
      if(toggleVisualise == true){
          int outcount = i/steps_out; // int divisible by modulo above
          std::string vtufilename;
          vtufilename = seedname+"_"+zero_pad_num(outcount)+".vtu";
          std::ofstream vtufile(vtufilename.c_str());
          lattice.outputSpinsVTU(vtufile);
          vtufile.close();
      }
    }
    physics->run(solver->getTime(),dt);
    solver->setTemperature(globalTemperature);
    solver->run();


  }
  double elapsed = static_cast<double>(std::clock()-start);
  elapsed/=CLOCKS_PER_SEC;
  output.write("Solving time: %f\n",elapsed);
  engfile.close();

  if(mag != NULL) { delete mag; }

}

void jams_finish() {
  using namespace globals;

  if(solver != NULL) { delete solver; }
  if(physics != NULL) { delete physics; }
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

  output.write("\n****************************************\n");
  output.write(  "               JAMS ERROR               \n");
  output.write(  "****************************************\n");
  output.write("%s\n",buffer);

  jams_finish();
  exit(EXIT_FAILURE);
}
