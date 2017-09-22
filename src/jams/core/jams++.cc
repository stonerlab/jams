// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#define JAMS_VERSION "1.5.0"
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#include <cstdarg>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>

#include "jams/core/config.h"
#include "jams/core/jams++.h"
#include "jams/core/load.h"
#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/types.h"
#include "jams/core/exception.h"
#include "jams/core/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/utils.h"
#include "jams/core/hamiltonian.h"

namespace {

  double dt = 0.0;
  int steps_min = 0;
  int steps_run = 0;
}  // anon namespace

int jams_initialize(int argc, char **argv) {
  output  = new Output();
  config  = new libconfig::Config();
  rng     = new Random();
  lattice = new Lattice();

  std::string config_filename;
  std::string config_patch_string;

  if (argc == 1) {
    jams_error("No config file specified");
  }

  config_filename = std::string(argv[1]);
  trim(config_filename);

  if (argc == 3) {
    config_patch_string = std::string(argv[2]);
  }


  seedname = file_basename(config_filename);
  trim(seedname);

  output->open("%s.out", seedname.c_str());

  output->write("\nJAMS++\n");
  output->write("Version %s\n", JAMS_VERSION);
  output->write("Commit %s\n", QUOTEME(GITCOMMIT));
  output->write("Compiled %s, %s\n", __DATE__, __TIME__);
  output->write("----------------------------------------\n");

  time_t rawtime;
  struct tm * timeinfo;
  char timebuffer[80];
  time(&rawtime);
  timeinfo = localtime(&rawtime); // NOLINT
  strftime(timebuffer, 80, "%b %d %Y, %X", timeinfo);
  output->write("Run time %s\n", timebuffer);
  output->write("----------------------------------------\n");

#ifdef DEBUG
  output->write("\nDEBUG Build\n");
#endif
  output->write("\nconfig file\n  %s\n", config_filename.c_str());

  {
    try {

      config->readFile(config_filename.c_str());

      if (!config_patch_string.empty()) {
        jams_patch_config(config_patch_string);

      }

      if (config->exists("sim.verbose")) {
        if (config->lookup("sim.verbose")) {
          output->enableVerbose();
          output->write("verbose output is ON\n");
        }
      }

//      jams_patch_config(argc, argv, config);

      unsigned int random_seed = time(NULL);
      if (config->lookupValue("sim.seed", random_seed)) {
        output->write("\nrandom seed in config file\n");
      } else {
        output->write("\nrandom seed from time\n");
      }
      output->write("  %u\n", random_seed);
      rng->seed(random_seed);


      dt = config->lookup("sim.t_step");
      output->write("\ntimestep\n  %1.8e\n", dt);

      double time_value = config->lookup("sim.t_run");
      steps_run = static_cast<int>(time_value/dt);
      output->write("\nruntime\n  %1.8e (%lu steps)\n",
        time_value, steps_run);

      if (config->exists("sim.t_min")) {
        time_value = config->lookup("sim.t_min");
        steps_min = static_cast<int>(time_value/dt);
      } else {
        steps_min = 0;
      }

      output->write("\nminimum runtime\n  %1.8e (%lu steps)\n",
        time_value, steps_min);

      lattice->init_from_config(*::config);

      output->write("\nInitialising physics module...\n");
      physics_module = Physics::create(config->lookup("physics"));

      output->write("\nInitialising solver...\n");
      solver = Solver::create(capitalize(config->lookup("sim.solver")));
      solver->initialize(argc, argv, dt);
      solver->register_physics_module(physics_module);

      if (!::config->exists("monitors")) {
        // no monitors were found in the config file - hopefully the physics module
        // produces the output!
        jams_warning("No monitors selected");
      } else {
        // loop over monitor groups and register
        const libconfig::Setting &monitor_settings = ::config->lookup("monitors");
        for (int i = 0; i != monitor_settings.getLength(); ++i) {
          solver->register_monitor(Monitor::create(monitor_settings[i]));
        }
      }

      if (!::config->exists("hamiltonians")) {
        jams_error("No hamiltonian terms selected");
      } else {
        // loop over hamiltonian groups and register
        const libconfig::Setting &hamiltonian_settings = ::config->lookup("hamiltonians");
        for (int i = 0; i != hamiltonian_settings.getLength(); ++i) {
          solver->register_hamiltonian(Hamiltonian::create(hamiltonian_settings[i], globals::num_spins));
        }
      }

      if(::config->exists("initializer")) {
        jams_global_initializer(::config->lookup("initializer"));
      }

    }
    catch(const libconfig::FileIOException &fioex) {
      jams_error("I/O error while reading '%s'", config_filename.c_str());
    }
    catch(const libconfig::ParseException &pex) {
      jams_error("Error parsing %s:%i: %s", pex.getFile(),
        pex.getLine(), pex.getError());
    }
    catch(const libconfig::SettingTypeException &stex) {
      jams_error("Config setting type error '%s'", stex.getPath());
    }
    catch(const libconfig::SettingNotFoundException &nfex) {
      jams_error("Required config setting not found '%s'", nfex.getPath());
    }
    catch(const general_exception &gex) {
      jams_error("%s", gex.what());
    }
#ifdef CUDA
    catch(const cuda_api_exception &cex) {
      jams_error("CUDA api exception\n '%s'", cex.what());
    }
#endif
    catch (std::exception& e) {
      jams_error("exception: %s", e.what());
    }
    catch(...) {
      jams_error("Caught an unknown exception");
    }
  }

  return 0;
}

void jams_run() {
  using namespace globals;

  output->write("\n----Data Run----\n");
  output->write("Running solver\n");
  std::clock_t start = std::clock();

  for (int i = 0; i < steps_run; ++i) {
    if (i > steps_min && solver->is_converged()) {
      break;
    }
    solver->update_physics_module();
    solver->notify_monitors();
    solver->run();
  }

  double elapsed = static_cast<double>(std::clock()-start);
  elapsed /= CLOCKS_PER_SEC;
  output->write("Solving time: %f\n", elapsed);
}

void jams_finish() {
  if (solver != NULL) { delete solver; }
  if (physics_module != NULL) { delete physics_module; }
}

void jams_error(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output->write("\n********************************************************************************\n\n");
  output->write("ERROR: %s\n\n", buffer);
  output->write("********************************************************************************\n\n");

  jams_finish();
  exit(EXIT_FAILURE);
}

void jams_warning(const char *string, ...) {
  va_list args;
  char buffer[1024];

  va_start(args, string);
  vsprintf(buffer, string, args);
  va_end(args);

  output->write("\n********************************************************************************\n\n");
  output->write("WARNING: %s\n\n", buffer);
  output->write("********************************************************************************\n\n");
}

void jams_global_initializer(const libconfig::Setting &settings) {
  if (settings.exists("spins")) {
    std::string file_name = settings["spins"];
    ::output->write("\nReading spin data from file: %s\n", file_name.c_str());
    load_array_from_file(file_name, "/spins", globals::s);
  }

  if (settings.exists("alpha")) {
    std::string file_name = settings["alpha"];
    ::output->write("\nReading alpha data from file: %s\n", file_name.c_str());
    load_array_from_file(file_name, "/alpha", globals::alpha);
  }

  if (settings.exists("mus")) {
    std::string file_name = settings["mus"];
    ::output->write("\nReading initial mus data from file: %s\n", file_name.c_str());
    load_array_from_file(file_name, "/mus", globals::mus);
  }

  if (settings.exists("gyro")) {
    std::string file_name = settings["gyro"];
    ::output->write("\nReading initial gyro data from file: %s\n", file_name.c_str());
    load_array_from_file(file_name, "/gyro", globals::gyro);
  }
}

void jams_patch_config(const std::string &patch_string) {
  libconfig::Config cfg_patch;

  try {
    cfg_patch.readFile(patch_string.c_str());
    ::output->write("patching form file\n  %s\n", patch_string.c_str());
  }
  catch(libconfig::FileIOException &fex) {
    cfg_patch.readString(patch_string);
    ::output->write("patching from string\n", patch_string.c_str());
    ::output->verbose("  %s\n", patch_string.c_str());
  }

  config_patch(::config->getRoot(), cfg_patch.getRoot());

  bool do_write_patched_config = true;

  config->lookupValue("sim.write_patched_config", do_write_patched_config);

  if (do_write_patched_config) {
    std::string patched_config_filename = seedname + "_patched.cfg";
    ::config->setFloatPrecision(8);
    ::config->writeFile(patched_config_filename.c_str());
  }
}