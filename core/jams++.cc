// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#define JAMS_VERSION "0.8.0"
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#include <cstdarg>
#include <fstream>
#include <iostream>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/monitor.h"
#include "core/physics.h"
#include "core/solver.h"
#include "core/utils.h"

#include "monitors/boltzmann.h"
#include "monitors/boltzmann_mag.h"
#include "monitors/energy.h"
#include "monitors/magnetisation.h"

#ifdef CUDA
#include <cublas.h>
#endif

namespace {
  Solver *solver;
  Physics *physics;
  double dt = 0.0;
  int steps_eq = 0;
  int steps_run = 0;
  int steps_out = 0;
  int  steps_vis = 0;
  int  steps_bin = 0;
  int steps_conv = 0;

  std::string convName;
  double convMeanTolerance = 0.0;
  double convDevTolerance = 0.0;

  bool toggleEnergy = false;
  bool toggleVisualise = false;
  bool toggleBinary = false;
  bool toggleSaveState = false;
  bool toggleReadState = false;
  bool toggleCoarse = false;
  std::string stateFileName;

  std::vector<Monitor*> monitor_list;
}  // anon namespace

int jams_init(int argc, char **argv) {
  if (argc == 1) {
        // seedname is executable
    seedname = std::string(argv[0]);
  } else {
        // seedname is first argument
    seedname = std::string(argv[1]);
  }
  trim(seedname);

  output.open("%s.out", seedname.c_str());

  output.write("\nJAMS++\n");
  output.write("Version %s\n", JAMS_VERSION);
  output.write("Commit %s\n", QUOTEME(GITCOMMIT));
  output.write("Compiled %s, %s\n", __DATE__, __TIME__);
  output.write("----------------------------------------\n");

  time_t rawtime;
  struct tm * timeinfo;
  char timebuffer[80];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(timebuffer, 80, "%b %d %Y, %X", timeinfo);
  output.write("Run time %s\n", timebuffer);
  output.write("----------------------------------------\n");

#ifdef DEBUG
  output.write("\nDEBUG Build\n");
#endif

  output.write("\nReading configuration file...\n");

  std::string cfgfile = seedname+".cfg";

  output.write("  * Config file: %s\n", cfgfile.c_str());

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


    double init_temperature = 0.0;

    try {
      dt = config.lookup("sim.t_step");
      output.write("  * Timestep:           %1.8e\n", dt);


      double tmp = config.lookup("sim.t_eq");
      steps_eq = static_cast<int>(tmp/dt);
      output.write("  * Equilibration time: %1.8e (%lu steps)\n",
        tmp, steps_eq);

      tmp = config.lookup("sim.t_run");
      steps_run = static_cast<int>(tmp/dt);
      output.write("  * Run time:           %1.8e (%lu steps)\n",
        tmp, steps_run);

      tmp = config.lookup("sim.t_out");
      steps_out = static_cast<int>(tmp/dt);
      output.write("  * Output time:        %1.8e (%lu steps)\n",
        tmp, steps_out);

      if (config.exists("sim.convergence") == true) {
        config.lookupValue("sim.convergence", convName);
        std::transform(solname.begin(), solname.end(),
          solname.begin(), toupper);
        config.lookupValue("sim.meanTolerance", convMeanTolerance);
        config.lookupValue("sim.devTolerance", convDevTolerance);

        tmp = config.lookup("sim.t_conv");
        steps_conv = static_cast<int>(tmp/dt);
        output.write("  * Convergence test time:        %1.8e (%lu steps)\n",
          tmp, steps_conv);
      }

      globals::h_app[0] = config.lookup("sim.h_app.[0]");
      globals::h_app[1] = config.lookup("sim.h_app.[1]");
      globals::h_app[2] = config.lookup("sim.h_app.[2]");

      if (config.exists("sim.read_state") == true) {
        config.lookupValue("sim.read_state", stateFileName);
        toggleReadState = true;
        output.write("  * Read state is ON\n");
      }

      if (config.exists("sim.save_state") == true) {
        config.lookupValue("sim.save_state", toggleSaveState);
        output.write("  * Save state is ON\n");
      }

      if (config.exists("sim.energy") == true) {
        config.lookupValue("sim.energy", toggleEnergy);
        if (toggleEnergy == true) {
          output.write("  * Energy calculation ON\n");
        } else {
          output.write("  * Energy calculation OFF\n");
        }
      }

      if (config.exists("sim.visualise") == true) {
        config.lookupValue("sim.visualise", toggleVisualise);
        if (toggleVisualise == true) {
          output.write("  * Visualisation is ON\n");
          tmp = config.lookup("sim.t_vis");
          steps_vis = static_cast<int>(tmp/dt);
          output.write("  * Visualisation time: %1.8e (%lu steps)\n",
            tmp, steps_vis);
        }
      } else {
        toggleVisualise = false;
      }

      if (config.exists("sim.binary") == true) {
        config.lookupValue("sim.binary", toggleBinary);
        if (toggleBinary == true) {
          output.write("  * Binary output is ON\n");
          tmp = config.lookup("sim.t_bin");
          steps_bin = static_cast<int>(tmp/dt);
          output.write("  * Binary output time: %1.8e (%lu steps)\n",
            tmp, steps_bin);
        }
      } else {
        toggleBinary = false;
      }

      if (config.exists("lattice.coarse") == true) {
        toggleCoarse = true;
        if (toggleCoarse == true) {
          output.write("  * Coarse magnetisation map output is ON\n");
        }
      } else {
        toggleCoarse = false;
      }

      if (config.exists("sim.seed") == true) {
        config.lookupValue("sim.seed", randomseed);
        output.write("  * Random generator seeded from config file\n");
      } else {
        randomseed = time(NULL);
        output.write("  * Random generator seeded from time\n");
      }
      output.write("  * Seed: %d\n", randomseed);

      init_temperature = config.lookup("sim.temperature");
      globals::globalTemperature = init_temperature;
      output.write("  * Initial temperature: %f\n", init_temperature);


      rng.seed(randomseed);

      lattice.createFromConfig(config);

      if (toggleBinary == true) {
        std::string binfilename;
        binfilename = seedname+"_types.bin";
        std::ofstream binfile(binfilename.c_str(),
          std::ios::binary|std::ios::out);
        lattice.outputTypesBinary(binfile);
        binfile.close();
      }

            // If read_state is true then attempt to read state from binary
            // file. If this fails (nspins != nspins in the file) then JAMS
            // exits with an error to avoid mistakingly thinking the file was
            // loaded. NOTE: This must be done after lattice is created but
            // before the solver is initialised so the GPU solvers can copy the
            // correct spin array.

      if (toggleReadState == true) {
        output.write("\nReading spin state from %s\n", stateFileName.c_str());
        std::ifstream statefile(stateFileName.c_str(),
          std::ios::binary|std::ios::in);
        lattice.readSpinsBinary(statefile);
        statefile.close();
      }


      if (config.exists("sim.solver") == true) {
        config.lookupValue("sim.solver", solname);
        std::transform(solname.begin(), solname.end(),
          solname.begin(), toupper);
      }



      output.write("\nInitialising physics module...\n");
      if (config.exists("sim.physics") == true) {
        config.lookupValue("sim.physics", physname);
        std::transform(physname.begin(), physname.end(),
          physname.begin(), toupper);

        if (physname == "FMR") {
          physics = Physics::Create(FMR);
        } else if (physname == "MFPT") {
          physics = Physics::Create(MFPT);
        } else if (physname == "TTM") {
          physics = Physics::Create(TTM);
        } else if (physname == "SPINWAVES") {
          physics = Physics::Create(SPINWAVES);
        } else if (physname == "DYNAMICSF") {
          physics = Physics::Create(DYNAMICSF);
        } else if (physname == "SQUARE") {
          physics = Physics::Create(SQUARE);
        } else if (physname == "FIELDCOOL") {
          physics = Physics::Create(FIELDCOOL);
        } else {
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
      jams_error("Setting '%s' not found", nfex.getPath());
    }
    catch(...) {
      jams_error("Undefined config error");
    }

    output.write("\nInitialising solver...\n");
    if (solname == "HEUNLLG") {
      solver = Solver::Create(HEUNLLG);
    } else if (solname == "CUDAHEUNLLG") {
      solver = Solver::Create(CUDAHEUNLLG);
    } else if (solname == "CUDASRK4LLG") {
      solver = Solver::Create(CUDASRK4LLG);
    } else if (solname == "METROPOLISMC") {
      solver = Solver::Create(METROPOLISMC);
    } else if (solname == "CUDAHEUNLLMS") {
      solver = Solver::Create(CUDAHEUNLLMS);
    } else if (solname == "CUDAHEUNLLBP") {
      solver = Solver::Create(CUDAHEUNLLBP);
    } else {  // default
      output.write("WARNING: Using default solver (HEUNLLG)\n");
      solver = Solver::Create();
    }

    solver->initialise(argc, argv, dt);
    solver->setTemperature(init_temperature);
  }

  // select monitors
  monitor_list.push_back(new MagnetisationMonitor());

  if (toggleEnergy == true) {
    monitor_list.push_back(new EnergyMonitor());
  }

  if ( config.exists("sim.monitors") == true ) {
    libconfig::Setting &simcfg = config.lookup("sim");

    for (int i = 0; i < simcfg["monitors"].getLength(); ++i) {
      std::string monname;
      monname = std::string(simcfg["monitors"][i].c_str());
      std::transform(monname.begin(), monname.end(),
        monname.begin(), toupper);

      if (monname == "BOLTZMANN") {
        monitor_list.push_back(new BoltzmannMonitor());
      } else {
        jams_error("Unknown monitor selected.");
      }
    }
  }

  for (int i = 0; i < monitor_list.size(); ++i) {
    monitor_list[i]->initialise();
  }

  if (convName == "MAG") {
    output.write("Convergence for Magnetisation\n");
    monitor_list[0]->initConvergence(convMag, convMeanTolerance,
      convDevTolerance);
  } else if (convName == "PHI") {
    output.write("Convergence for Phi\n");
    monitor_list[0]->initConvergence(convPhi, convMeanTolerance,
      convDevTolerance);
  } else if (convName == "SINPHI") {
    output.write("Convergence for Sin(Phi)\n");
    monitor_list[0]->initConvergence(convSinPhi, convMeanTolerance,
      convDevTolerance);
  }
  output.write("StdDev Tolerance: %e\n", convMeanTolerance,
    convDevTolerance);



  return 0;
}

void jams_run() {
  using namespace globals;

  std::ofstream coarseFile;

  if (toggleCoarse == true) {
    std::string coarseFileName;
    coarseFileName = seedname+"_map.dat";
    coarseFile.open(coarseFileName.c_str());
  }


  globalSteps = 0;
  output.write("\n----Equilibration----\n");
  output.write("Running solver\n");
  for (int i = 0; i < steps_eq; ++i) {
    if (i%steps_out == 0) {
      solver->syncOutput();

      monitor_list[0]->write(solver);
    }
    physics->run(solver->getTime(), dt);
    solver->setTemperature(globalTemperature);
    solver->run();
    globalSteps++;
  }

  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  std::clock_t start = std::clock();
  for (int i = 0; i < steps_run; ++i) {
    if (i%steps_out == 0) {
      solver->syncOutput();
      for (int i = 0; i < monitor_list.size(); ++i) {
        monitor_list[i]->write(solver);
      }
      physics->monitor(solver->getTime(), dt);

      if (toggleCoarse == true) {
        lattice.outputCoarseMagnetisationMap(coarseFile);
        coarseFile << "\n\n";
      }
    }
    if (toggleVisualise == true) {
      if (i%steps_vis == 0) {
        int outcount = i/steps_vis;  // int divisible by modulo above
        std::string vtufilename;
        vtufilename = seedname+"_"+zero_pad_num(outcount)+".vtu";
        std::ofstream vtufile(vtufilename.c_str());
        lattice.outputSpinsVTU(vtufile);
        vtufile.close();
      }
    }

    if (toggleBinary == true) {
      if (i%steps_bin == 0) {
        int outcount = i/steps_bin;  // int divisible by modulo above
        std::string binfilename;
        binfilename = seedname+"_"+zero_pad_num(outcount)+".bin";
        std::ofstream binfile(binfilename.c_str(),
          std::ios::binary|std::ios::out);
        lattice.outputSpinsBinary(binfile);
        binfile.close();
      }
    }

    if (steps_conv > 0) {
      if ((i+1)%steps_conv == 0) {
        if (monitor_list[0]->checkConvergence() == true) {
          break;
        }
      }
    }

    physics->run(solver->getTime(), dt);
    solver->setTemperature(globalTemperature);
    solver->run();
    globalSteps++;
    for (int i = 0; i < monitor_list.size(); ++i) {
      monitor_list[i]->run();
    }
  }

  if (toggleSaveState == true) {
    output.write(
      "\n-------------------\nSaving spin state\n-------------------\n");
    std::string statefilename = seedname+"_state.dat";
    std::ofstream statefile(statefilename.c_str(),
      std::ios::out|std::ios::binary|std::ios::trunc);
    lattice.outputSpinsBinary(statefile);
    statefile.close();
  }

  double elapsed = static_cast<double>(std::clock()-start);
  elapsed /= CLOCKS_PER_SEC;
  output.write("Solving time: %f\n", elapsed);

  for (int i = 0; i < monitor_list.size(); ++i) {
    if (monitor_list[i] != NULL) {
      delete monitor_list[i];
      monitor_list[i] = NULL;
    }
  }

  if (toggleCoarse == true) {
    coarseFile.close();
  }
}

void jams_finish() {
  if (solver != NULL) { delete solver; }
  if (physics != NULL) { delete physics; }
}

int main(int argc, char **argv) {
  jams_init(argc, argv);

  jams_run();

  jams_finish();
  return EXIT_SUCCESS;
}

void jams_error(const char *string, ...) {
  // TODO(joe): Fix this so that the arguments are passed through.
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output.write("\n****************************************\n");
  output.write("               JAMS ERROR               \n");
  output.write("****************************************\n");
  output.write("%s\n", buffer);

  jams_finish();
  exit(EXIT_FAILURE);
}
