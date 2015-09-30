// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#define JAMS_VERSION "1.0.0"
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/monitor.h"
#include "core/physics.h"
#include "core/solver.h"
#include "core/utils.h"
#include "core/hamiltonian.h"

#ifdef CUDA
#include <cublas.h>
#endif

namespace {

  double dt = 0.0;
  int steps_run = 0;
  int  steps_bin = 0;

  bool energy_output_is_set = false;
  bool binary_output_is_set = false;
  bool coarse_output_is_set = false;

  bool save_state_is_set = false;

}  // anon namespace

int jams_initialize(int argc, char **argv) {
  std::string config_filename;

  if (argc == 1) {
    jams_error("No config file specified");
  } else {
    // config file is the only argument
    config_filename = std::string(argv[1]);
    trim(config_filename);
  }

  seedname = file_basename(config_filename);
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
  timeinfo = localtime(&rawtime); // NOLINT
  strftime(timebuffer, 80, "%b %d %Y, %X", timeinfo);
  output.write("Run time %s\n", timebuffer);
  output.write("----------------------------------------\n");

#ifdef DEBUG
  output.write("\nDEBUG Build\n");
#endif

  output.write("\nReading configuration file...\n");

  output.write("  * Config file: %s\n", config_filename.c_str());

  {
    try {
      config.readFile(config_filename.c_str());

      verbose_output_is_set = false;
      config.lookupValue("sim.verbose", verbose_output_is_set);
      if (verbose_output_is_set) {
        output.enableVerbose();
        output.write("  * Verbose output is ON\n");
      }


      ::optimize::use_fft = false;
      ::config.lookupValue("sim.fft", ::optimize::use_fft);
      if (::optimize::use_fft) {
        output.write("  * FFT optimizations have been requested (not guaranteed)\n");
      }

      dt = config.lookup("sim.t_step");
      output.write("  * Timestep:           %1.8e\n", dt);

      double time_value = config.lookup("sim.t_eq");

      time_value = config.lookup("sim.t_run");
      steps_run = static_cast<int>(time_value/dt);
      output.write("  * Run time:           %1.8e (%lu steps)\n",
        time_value, steps_run);

      if (config.lookupValue("sim.save_state", save_state_is_set)) {
        if (save_state_is_set) {
          output.write("  * Save state is ON\n");
        }
      }

      if (config.lookupValue("sim.energy", energy_output_is_set)) {
        if (energy_output_is_set) {
          output.write("  * Energy calculation ON\n");
        } else {
          output.write("  * Energy calculation OFF\n");
        }
      }

      if (config.lookupValue("sim.binary", binary_output_is_set)) {
        if (binary_output_is_set) {
          output.write("  * Binary output is ON\n");
          time_value = config.lookup("sim.t_bin");
          steps_bin = static_cast<int>(time_value/dt);
          output.write("  * Binary output time: %1.8e (%lu steps)\n",
            time_value, steps_bin);
        }
      }

      if (config.exists("lattice.coarse")) {
        coarse_output_is_set = true;
        output.write("  * Coarse magnetisation map output is ON\n");
      }

      unsigned int randomseed;
      if (config.lookupValue("sim.seed", randomseed)) {
        output.write("  * Random generator seeded from config file\n");
      } else {
        randomseed = time(NULL);
        output.write("  * Random generator seeded from time\n");
      }
      output.write("  * Seed: %u\n", randomseed);

      rng.seed(randomseed);

      lattice.init_from_config(::config);

      output.write("\nInitialising physics module...\n");
      physics_module = Physics::create(config.lookup("physics"));

      output.write("\nInitialising solver...\n");
      solver = Solver::create(capitalize(config.lookup("sim.solver")));
      solver->initialize(argc, argv, dt);
      solver->register_physics_module(physics_module);

      if (!::config.exists("monitors")) {
        // no monitors were found in the config file - hopefully the physics module
        // produces the output!
        jams_warning("No monitors selected");
      } else {
        // loop over monitor groups and register
        const libconfig::Setting &monitor_settings = ::config.lookup("monitors");
        for (int i = 0; i != monitor_settings.getLength(); ++i) {
          solver->register_monitor(Monitor::create(monitor_settings[i]));
        }
      }

      if (!::config.exists("hamiltonians")) {
        jams_error("No hamiltonian terms selected");
      } else {
        // loop over hamiltonian groups and register
        const libconfig::Setting &hamiltonian_settings = ::config.lookup("hamiltonians");
        for (int i = 0; i != hamiltonian_settings.getLength(); ++i) {
          solver->register_hamiltonian(Hamiltonian::create(hamiltonian_settings[i]));
        }
      }

    }
    catch(const libconfig::FileIOException &fioex) {
      jams_error("I/O error while reading '%s'", config_filename.c_str());
    }
    catch(const libconfig::ParseException &pex) {
      jams_error("Error parsing %s:%i: %s", pex.getFile(),
        pex.getLine(), pex.getError());
    }
    catch(const libconfig::SettingNotFoundException &nfex) {
      jams_error("Required config setting not found '%s'", nfex.getPath());
    }
    catch (std::exception& e) {
      jams_error("Error: %s", e.what());
    }
    catch(...) {
      jams_error("Caught an unknown exception");
    }
  }

  return 0;
}

void jams_run() {
  using namespace globals;

  std::ofstream coarse_magnetisation_file;

  if (coarse_output_is_set) {
    coarse_magnetisation_file.open(std::string(seedname+"_map.dat").c_str());
  }

  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  std::clock_t start = std::clock();

  for (int i = 0; i < steps_run; ++i) {
    if (i > 1000 && solver->is_converged()) {
      break;
    }
    solver->update_physics_module();
    solver->notify_monitors();
    solver->run();
  }

  double elapsed = static_cast<double>(std::clock()-start);
  elapsed /= CLOCKS_PER_SEC;
  output.write("Solving time: %f\n", elapsed);

  if (coarse_output_is_set) {
    coarse_magnetisation_file.close();
  }
}

void jams_finish() {
  if (solver != NULL) { delete solver; }
  if (physics_module != NULL) { delete physics_module; }
}

int main(int argc, char **argv) {
  jams_initialize(argc, argv);

  jams_run();

  jams_finish();
  return EXIT_SUCCESS;
}

void jams_error(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output.write("\n********************************************************************************\n\n");
  output.write("ERROR: %s\n\n", buffer);
  output.write("********************************************************************************\n\n");

  jams_finish();
  exit(EXIT_FAILURE);
}

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output.write("\n********************************************************************************\n\n");
  output.write("WARNING: %s\n\n", buffer);
  output.write("********************************************************************************\n\n");
}
