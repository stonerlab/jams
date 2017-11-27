// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#include "version.h"

#include <cstdarg>
#include <fstream>

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/jams++.h"
#include "jams/core/lattice.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"
#include "jams/core/rand.h"
#include "jams/core/solver.h"
#include "jams/helpers/duration.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/load.h"
#include "jams/interface/config.h"

using namespace std;

namespace jams {

    void new_global_classes() {
      config = new libconfig::Config();
      rng = new Random();
      lattice = new Lattice();
    }

    void delete_global_classes() {
      delete solver;
      delete lattice;
      delete rng;
      delete config;
    }

    void process_command_line_args(int argc, char **argv, jams::Simulation &sim) {
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

    std::string section(const std::string& name) {
      std::string line = "\n--------------------------------------------------------------------------------\n\n";
      return line.replace(1, name.size() + 1, name + " ");
    }
}

void jams_initialize(int argc, char **argv) {
  std::cin.tie(nullptr);
  ios_base::sync_with_stdio(false);

  jams::Simulation simulation;

  jams::new_global_classes();

  cout << "\nJAMS++ " << jams::build::version << "\n\n";
  cout << "build   ";
  cout << jams::build::time << " ";
  cout << jams::build::type << " ";
  cout << jams::build::hash << " ";
  cout << jams::build::branch << "\n";
  cout << "run     ";
  cout << get_date_string(std::chrono::system_clock::now()) << "\n";

  jams::process_command_line_args(argc, argv, simulation);
  seedname = simulation.name;

  // TODO: tee cout also to a log file
  cout << "config  " << simulation.config_file_name << "\n";

  try {

    config->readFile(simulation.config_file_name.c_str());

    if (!simulation.config_patch_string.empty()) {
      jams_patch_config(simulation.config_patch_string);
    }

    if (config->exists("sim")) {
      simulation.verbose = jams::config_optional<bool>(config->lookup("sim"), "verbose", false);
      simulation.random_seed = jams::config_optional<int>(config->lookup("sim"), "seed", simulation.random_seed);
    }

    cout << "verbose " << simulation.verbose << "\n";
    cout << "seed    " << simulation.random_seed << "\n";

    rng->seed(static_cast<const uint32_t>(simulation.random_seed));

    cout << jams::section("init lattice");

    lattice->init_from_config(*::config);

    cout << jams::section("init solver");

    solver = Solver::create(config->lookup("solver"));
    solver->initialize(config->lookup("solver"));
    // todo: fix this memory leak
    solver->register_physics_module(Physics::create(config->lookup("physics")));

    cout << jams::section("init monitors");

    if (!::config->exists("monitors")) {
      jams_warning("No monitors in config");
    } else {
      const libconfig::Setting &monitor_settings = ::config->lookup("monitors");
      for (auto i = 0; i < monitor_settings.getLength(); ++i) {
        solver->register_monitor(Monitor::create(monitor_settings[i]));
      }
    }

    cout << jams::section("init hamiltonians");

    if (!::config->exists("hamiltonians")) {
      jams_error("No hamiltonians in config");
    } else {
      const libconfig::Setting &hamiltonian_settings = ::config->lookup("hamiltonians");
      for (auto i = 0; i < hamiltonian_settings.getLength(); ++i) {
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

void jams_run() {
  using namespace std::chrono;

  cout << jams::section("running solver");
  auto start_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "start   " << get_date_string(start_time) << "\n\n";

  while (::solver->is_running()) {
    if (::solver->is_converged()) {
      break;
    }

    ::solver->update_physics_module();
    ::solver->notify_monitors();
    ::solver->run();
  }

  auto end_time = time_point_cast<milliseconds>(system_clock::now());
  cout << "finish  " << get_date_string(end_time) << "\n\n";
  cout << "runtime " << duration_string(end_time - start_time) << "\n";
}

void jams_finish() {
  jams::delete_global_classes();
}

void jams_global_initializer(const libconfig::Setting &settings) {
  if (settings.exists("spins")) {
    std::string file_name = settings["spins"];
    cout << "reading spin data from file " << file_name << "\n";
    load_array_from_file(file_name, "/spins", globals::s);
  }

  if (settings.exists("alpha")) {
    std::string file_name = settings["alpha"];
    cout << "reading alpha data from file " << file_name << "\n";
    load_array_from_file(file_name, "/alpha", globals::alpha);
  }

  if (settings.exists("mus")) {
    std::string file_name = settings["mus"];
    cout << "reading mus data from file " << file_name << "\n";
    load_array_from_file(file_name, "/mus", globals::mus);
  }

  if (settings.exists("gyro")) {
    std::string file_name = settings["gyro"];
    cout << "reading gyro data from file " << file_name << "\n";
    load_array_from_file(file_name, "/gyro", globals::gyro);
  }
}

void jams_patch_config(const std::string &patch_string) {
  libconfig::Config cfg_patch;

  try {
    cfg_patch.readFile(patch_string.c_str());
    cout << "patching form file " << patch_string << "\n";
  }
  catch(libconfig::FileIOException &fex) {
    cfg_patch.readString(patch_string);
    cout << "patching from string " << patch_string << "\n";
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