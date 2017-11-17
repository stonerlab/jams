// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN

#include <cstdarg>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "version.h"
#include "jams/interface/config.h"
#include "jams/core/jams++.h"
#include "jams/helpers/load.h"
#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/types.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/error.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/duration.h"
#include "hamiltonian.h"

namespace jams {

    void new_global_classes() {
      output  = new Output();
      config  = new libconfig::Config();
      rng     = new Random();
      lattice = new Lattice();
    }

    void delete_global_classes() {
      delete solver;
      delete physics_module;
      delete lattice;
      delete rng;
      delete config;
      delete output;
    }

    void process_command_line_args(int argc, char **argv, jams::Simulation& sim) {
      if (argc == 1) {
        jams_error("No config file specified");
      }

      sim.config_file_name = std::string(argv[1]);
      trim(sim.config_file_name);

      if (argc == 3) {
        sim.config_patch_string = std::string(argv[2]);
      }

      sim.name = file_basename(sim.config_file_name);
      trim(sim.name);

      sim.log_file_name = sim.name + ".log";
    }
}

int jams_initialize(int argc, char **argv) {

  jams::Simulation simulation;

  jams::new_global_classes();

  output->write("\nJAMS++ %s\n\n", VERSION);

  output->write("build   %s %s %s %s\n", BUILD_TIME, BUILD_TYPE, GIT_COMMIT_HASH, GIT_BRANCH);
  output->write("run     %s\n", get_date_string(std::chrono::system_clock::now()).c_str());

  jams::process_command_line_args(argc, argv, simulation);
  seedname = simulation.name;

  output->open("%s", simulation.log_file_name.c_str());
  output->write("log     %s\n", simulation.log_file_name.c_str());
  output->write("config  %s\n", simulation.config_file_name.c_str());

  {
    try {

      config->readFile(simulation.config_file_name.c_str());

      if (!simulation.config_patch_string.empty()) {
        jams_patch_config(simulation.config_patch_string);
      }

      if (config->exists("sim")) {
        if (config->lookup("sim")) {
          simulation.verbose = true;
          output->enableVerbose();
        }

        simulation.random_seed = jams::config_optional<int>(config->lookup("sim"), "seed", simulation.random_seed);
      }


      output->write("verbose %s\n", simulation.verbose ? "true" : "false");
      output->write("seed    %d\n", simulation.random_seed);

      rng->seed(static_cast<const uint32_t>(simulation.random_seed));

      output->write("\ninit lattice -------------------------------------------------------------------\n\n");

      lattice->init_from_config(*::config);

      output->write("\ninit physics -------------------------------------------------------------------\n\n");

      physics_module = Physics::create(config->lookup("physics"));

      output->write("\ninit solver --------------------------------------------------------------------\n\n");

      solver = Solver::create(config->lookup("solver"));
      solver->initialize(config->lookup("solver"));
      solver->register_physics_module(physics_module);

      output->write("\ninit monitors ------------------------------------------------------------------\n\n");

      if (!::config->exists("monitors")) {
        jams_warning("No monitors in config");
      } else {
        const libconfig::Setting &monitor_settings = ::config->lookup("monitors");
        for (int i = 0; i < monitor_settings.getLength(); ++i) {
          solver->register_monitor(Monitor::create(monitor_settings[i]));
        }
      }

      output->write("\ninit hamiltonians --------------------------------------------------------------\n\n");

      if (!::config->exists("hamiltonians")) {
        jams_error("No hamiltonians in config");
      } else {
        const libconfig::Setting &hamiltonian_settings = ::config->lookup("hamiltonians");
        for (int i = 0; i < hamiltonian_settings.getLength(); ++i) {
          solver->register_hamiltonian(Hamiltonian::create(hamiltonian_settings[i], globals::num_spins));
        }
      }

      if(::config->exists("initializer")) {
        jams_global_initializer(::config->lookup("initializer"));
      }
    }
    catch(const libconfig::FileIOException &fioex) {
      jams_error("I/O error while reading '%s'", simulation.config_file_name.c_str());
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
  using namespace std::chrono;

  output->write("\nrunning solver -----------------------------------------------------------------\n\n");
  output->write("start   %s\n\n", get_date_string(system_clock::now()).c_str());

  auto start_time = time_point_cast<milliseconds>(system_clock::now());

  while (solver->is_running()) {
    if (solver->is_converged()) {
      break;
    }

    solver->update_physics_module();
    solver->notify_monitors();
    solver->run();
  }

  output->write("finish  %s\n\n", get_date_string(system_clock::now()).c_str());

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  output->write("runtime %s\n", duration_string(end_time - start_time).c_str());
}

void jams_finish() {
  jams::delete_global_classes();
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

#if (((LIBCONFIG_VER_MAJOR == 1) && (LIBCONFIG_VER_MINOR >= 6)) \
     || (LIBCONFIG_VER_MAJOR > 1))
    ::config->setFloatPrecision(8);
#endif

    ::config->writeFile(patched_config_filename.c_str());
  }
}