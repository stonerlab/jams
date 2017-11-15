// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#define JAMS_VERSION "1.5.0"
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#include <cstdarg>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>

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

    void process_command_line_args(int argc, char **argv,
                                   std::string &config_filename,
                                   std::string &config_patch_string) {
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
    }

    void output_program_header() {
      output->write("\nJAMS++\n");
      output->write("Version %s\n", JAMS_VERSION);
      output->write("Commit %s\n", QUOTEME(GITCOMMIT));
      output->write("Compiled %s, %s\n", __DATE__, __TIME__);
      output->write("----------------------------------------\n");
      output->write("Run time %s\n", get_date_string(std::chrono::system_clock::now()).c_str());
      output->write("----------------------------------------\n");
#ifdef DEBUG
      output->write("\nDEBUG Build\n");
#endif
    }
}

int jams_initialize(int argc, char **argv) {

  jams::new_global_classes();
  jams::output_program_header();

  std::string config_filename;
  std::string config_patch_string;

  jams::process_command_line_args(argc, argv, config_filename, config_patch_string);

  output->open("%s.out", seedname.c_str());

  output->write("\nconfig file\n  %s\n", config_filename.c_str());

  {
    try {

      config->readFile(config_filename.c_str());

      if (!config_patch_string.empty()) {
        jams_patch_config(config_patch_string);
      }

      auto verbose_output = jams::config_optional<bool>(config->lookup("sim"), "verbose", jams::default_sim_verbose_output);
      if (verbose_output) {
        output->enableVerbose();
        output->write("verbose output is ON\n");
      }

      auto random_seed = jams::config_optional<int>(config->lookup("sim"), "seed", time(nullptr));
      output->write("  %d\n", random_seed);
      rng->seed(static_cast<const uint32_t>(random_seed));

      lattice->init_from_config(*::config);

      output->write("\nInitialising physics module...\n");
      physics_module = Physics::create(config->lookup("physics"));

      output->write("\nInitialising solver...\n");
      solver = Solver::create(config->lookup("solver"));
      solver->initialize(config->lookup("solver"));
      solver->register_physics_module(physics_module);

      if (!::config->exists("monitors")) {
        jams_warning("No monitors in config");
      } else {
        const libconfig::Setting &monitor_settings = ::config->lookup("monitors");
        for (int i = 0; i < monitor_settings.getLength(); ++i) {
          solver->register_monitor(Monitor::create(monitor_settings[i]));
        }
      }

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

  auto start_time = std::clock();

  while (solver->is_running()) {
    if (solver->is_converged()) {
      break;
    }

    solver->update_physics_module();
    solver->notify_monitors();
    solver->run();
  }

  auto end_time = std::clock();
  output->write("Solving time: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);
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