// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#include "version.h"

#include <cstdarg>
#include <fstream>

#if HAS_OMP
  #include <omp.h>
#endif

#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/jams++.h"
#include "jams/core/lattice.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"
#include "jams/helpers/random.h"
#include "jams/core/solver.h"
#include "jams/helpers/duration.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/load.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"
#include "jams/helpers/timer.h"
#include "jams/helpers/progress_bar.h"
#include "jams/containers/multiarray.h"

using namespace std;

namespace jams {

    void new_global_classes() {
      config = new libconfig::Config();
      lattice = new Lattice();
    }

    void delete_global_classes() {
      delete solver;
      delete lattice;
      delete config;
    }

    void parse_args(int argc, char **argv, jams::Simulation &sim) {
      if (argc == 1) jams_die("No config file specified");

      sim.config_file_name    = trim(string(argv[1]));
      sim.config_patch_string = (argc == 3 ? string(argv[2]) : "");
      sim.name                = trim(file_basename(sim.config_file_name));
    }

    void parse_config(jams::Simulation &sim) {
      try {
        config->readFile(sim.config_file_name.c_str());
      }
      catch(const libconfig::FileIOException &fioex) {
        jams_die("I/O error while reading '%s'", sim.config_file_name.c_str());
      }
      catch(const libconfig::ParseException &pex) {
        jams_die("Error parsing %s:%i: %s", pex.getFile(),
                 pex.getLine(), pex.getError());
      }

      jams_patch_config(sim.config_patch_string);
    }

    std::string section(const std::string& name) {
      std::string line = "\n--------------------------------------------------------------------------------\n";
      return line.replace(1, name.size() + 1, name + " ");
    }

    string header() {
      stringstream ss;
      ss << "\nJAMS++ " << jams::build::version << "\n\n";
      ss << "build   "  << jams::build::type << " (";
        ss << "cuda:" << jams::build::option_cuda;
        ss << " omp:" << jams::build::option_omp;
        ss << " fastmath:" << jams::build::option_fastmath ;
        ss << " mixed_prec:" << jams::build::option_mixed_prec << ")\n";
      ss << "git     "  << jams::build::branch << " (" << jams::build::hash << ")\n" ;
      ss << "run     ";
      ss << get_date_string(std::chrono::system_clock::now()) << "\n";
      #if HAS_OMP
      ss << "threads " << omp_get_max_threads() << "\n";
      #endif
      return ss.str();
    }
}

void jams_initialize(int argc, char **argv) {
  jams::desync_io();
  cout << jams::header();

  jams::Simulation simulation;
  jams::new_global_classes();
  jams::parse_args(argc, argv, simulation);
  seedname = simulation.name;
  cout << "config  " << simulation.config_file_name << "\n";   // TODO: tee cout also to a log file

  jams::parse_config(simulation);

  simulation.random_state = jams::random_generator_internal_state();

  try {
    ::config->setAutoConvert(true);
    if (::config->exists("sim")) {
      simulation.verbose = jams::config_optional<bool>(config->lookup("sim"), "verbose", false);

      if (config->exists("sim.seed")) {
        simulation.random_seed = jams::config_required<unsigned long>(config->lookup("sim"), "seed");
        jams::random_generator().seed(simulation.random_seed);
        cout << "seed    "   << simulation.random_seed << "\n";
      }

      if (config->exists("sim.rng_state")) {
        auto state = jams::config_required<string>(config->lookup("sim"), "rng_state");
        istringstream(state) >> simulation.random_state;
      }
    }

    cout << "verbose "   << simulation.verbose << "\n";
    cout << "rng state " << simulation.random_state << "\n";

    cout << jams::section("init lattice") << std::endl;

    lattice->init_from_config(*::config);

    cout << jams::section("init solver") << std::endl;

    solver = Solver::create(config->lookup("solver"));
    solver->initialize(config->lookup("solver"));
    solver->register_physics_module(Physics::create(config->lookup("physics")));     // todo: fix this memory leak

    cout << jams::section("init hamiltonians") << std::endl;

    if (!::config->exists("hamiltonians")) {
      jams_die("No hamiltonians in config");
    } else {
      const libconfig::Setting &hamiltonian_settings = ::config->lookup("hamiltonians");
      for (auto i = 0; i < hamiltonian_settings.getLength(); ++i) {
        solver->register_hamiltonian(Hamiltonian::create(hamiltonian_settings[i], globals::num_spins, solver->is_cuda_solver()));
      }
    }

    cout << jams::section("init monitors") << std::endl;

    if (!::config->exists("monitors")) {
      jams_warning("No monitors in config");
    } else {
      const libconfig::Setting &monitor_settings = ::config->lookup("monitors");
      for (auto i = 0; i < monitor_settings.getLength(); ++i) {
        solver->register_monitor(Monitor::create(monitor_settings[i]));
      }
    }

    if(::config->exists("initializer")) {
      jams_global_initializer(::config->lookup("initializer"));
    }
  }
  catch(const libconfig::SettingTypeException &stex) {
    jams_die("Config setting type error '%s'", stex.getPath());
  }
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_die("Required config setting not found '%s'", nfex.getPath());
  }
  catch(const jams::runtime_error &gex) {
    jams_die("%s", gex.what());
  }
  catch (std::exception& e) {
    jams_die("exception: %s", e.what());
  }
  catch(...) {
    jams_die("Caught an unknown exception");
  }
}

void jams_run() {
  try {
    cout << jams::section("running solver") << std::endl;
    cout << "start   " << get_date_string(std::chrono::system_clock::now()) << "\n" << std::endl;
    {
      ProgressBar progress;
      Timer<> timer;
      while (::solver->is_running()) {
        if (::solver->is_converged()) {
          break;
        }

        ::solver->update_physics_module();
        ::solver->notify_monitors();
        ::solver->run();

        progress.set(double(::solver->iteration()) / double(::solver->max_steps()));
        if (::solver->iteration() % 1000 == 0) {
          cout << progress;
        }
      }
      cout << "\n\n";
      cout << "runtime " << timer.elapsed_time() << " seconds" << std::endl;

      cout << "finish  " << get_date_string(std::chrono::system_clock::now()) << "\n\n";
    }

    {
      cout << jams::section("running post process") << std::endl;
      cout << "start   " << get_date_string(std::chrono::system_clock::now()) << "\n" << std::endl;

      Timer<> timer;

      for (auto m : solver->monitors()) {
        m->post_process();
      }
      cout << "runtime " << timer.elapsed_time() << " seconds" << std::endl;
      cout << "finish  " << get_date_string(std::chrono::system_clock::now()) << "\n\n";
    }

  }
  catch(const jams::runtime_error &gex) {
    jams_die("%s", gex.what());
  }
  catch (std::exception& e) {
    jams_die("exception: %s", e.what());
  }
  catch(...) {
    jams_die("Caught an unknown exception");
  }
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

  if (patch_string.empty()) return;

  try {
    cfg_patch.readFile(patch_string.c_str());
    cout << "patching form file " << patch_string << "\n";
  }
  catch(libconfig::FileIOException &fex) {
    cfg_patch.readString(patch_string);
    cout << "patching from string " << patch_string << "\n";
  }
  catch(const libconfig::ParseException &pex) {
    jams_die("Error parsing %s:%i: %s", pex.getFile(),
             pex.getLine(), pex.getError());
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