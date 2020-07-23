// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#include "version.h"

#include <fstream>
#include <jams/common.h>

#if HAS_OMP
  #include <omp.h>
#endif

#include "jams/core/args.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/jams++.h"
#include "jams/core/lattice.h"
#include "jams/core/monitor.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/duration.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/load.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/timer.h"
#include "jams/helpers/progress_bar.h"

using namespace std;

namespace jams {

    void new_global_classes() {
      lattice = new Lattice();
    }

    void delete_global_classes() {
      delete solver;
      delete lattice;
    }

    // Reads a vector of strings in order, combining to produce a config.
    //
    // If the string is an existent file name it is loaded as a config,
    // otherwise it is directly interpreted as a config string.
    void parse_config_strings(const vector<string>& config_strings, unique_ptr<libconfig::Config>& combined_config) {
      if (!combined_config) {
        combined_config.reset(new libconfig::Config);
      }

      for (const auto &s : config_strings) {
        libconfig::Config patch;
        if (jams::system::file_exists(s)) {
          try {
            patch.readFile(s.c_str());
          }
          catch (libconfig::FileIOException &fex) {
            throw std::runtime_error("IO error opening config file: " + s);
          }
          catch (const libconfig::ParseException &pex) {
            throw std::runtime_error("Error parsing config file: "
                                     + string(pex.getFile()) + ":"
                                     + to_string(pex.getLine()) + ":"
                                     + string(pex.getError()));
          }
        } else {
          try {
            patch.readString(s.c_str());
          }
          catch (const libconfig::ParseException &pex) {
            stringstream ss;
            ss << "File not found or error parsing config string:\n";
            ss << "  '" << s << "'\n";
            ss << "line " << to_string(pex.getLine()) << ": " << string(pex.getError());

            throw std::runtime_error(ss.str());
          }
        }

        overwrite_config_settings(combined_config->getRoot(), patch.getRoot());
      }
    }

    void write_config(const std::string& filename, const unique_ptr<libconfig::Config> &cfg) {
      cfg->setFloatPrecision(jams::defaults::config_float_precision);
      cfg->writeFile(filename.c_str());
    }

    void set_mode() {
      std::string solver_name = config->lookup("solver.module");
      if (contains(lowercase(solver_name), "gpu")) {
        jams::instance().set_mode(Mode::GPU);
        if (!jams::instance().has_gpu_device()) {
          throw std::runtime_error("No CUDA device available");
        }
      } else {
        jams::instance().set_mode(Mode::CPU);
      }
    }

    std::string section(const std::string &name) {
      std::string line = "\n--------------------------------------------------------------------------------\n";
      return line.replace(1, name.size() + 1, name + " ");
    }

    string build_info() {
      stringstream ss;
      ss << "  commit     " << jams::build::hash << "\n";
      ss << "  branch     " << jams::build::branch << "\n";
      ss << "  build      " << jams::build::type << "\n";
      ss << "  cuda       " << jams::build::option_cuda << "\n";
      ss << "  omp        " << jams::build::option_omp << "\n";
      ss << "  fastmath   " << jams::build::option_fastmath << "\n";
      ss << "  mixed_prec " << jams::build::option_mixed_prec << "\n";
      ss << "  libconfig  " << jams::build::libconfig_version << "\n";
      ss << "    " << find_and_replace(jams::build::libconfig_libraries, ";", "\n    ") << "\n";
      ss << "  highfive   " << jams::build::highfive_version << "\n";
      ss << "    " << find_and_replace(jams::build::highfive_libraries, ";", "\n    ") << "\n";
      ss << "  spglib     " << jams::build::spglib_version << "\n";
      ss << "    " << find_and_replace(jams::build::spglib_libraries, ";", "\n    ") << "\n";
      ss << "  pcg        " << jams::build::pcg_version << "\n";
      ss << "    " << find_and_replace(jams::build::pcg_libraries, ";", "\n    ") << "\n";
      ss << "  hdf5       " << jams::build::hdf5_version << "\n";
      ss << "    " << find_and_replace(jams::build::hdf5_libraries, ";", "\n    ") << "\n";
      #if HAS_MKL
      ss << "  mkl        " << jams::build::mkl_version() << "\n";
      ss << "    " << find_and_replace(jams::build::mkl_libraries, ";", "\n    ") << "\n";
      #endif
      #if HAS_CUDA
      ss << "  cusparse   " << "\n";
      ss << "    " << find_and_replace(jams::build::cusparse_libraries, ";", "\n    ") << "\n";
      ss << "  curand   " << "\n";
      ss << "    " << find_and_replace(jams::build::curand_libraries, ";", "\n    ") << "\n";
      ss << "  cublas   " << "\n";
      ss << "    " << find_and_replace(jams::build::cublas_libraries, ";", "\n    ") << "\n";
      ss << "  cufft   " << "\n";
      ss << "    " << find_and_replace(jams::build::cufft_libraries, ";", "\n    ") << "\n";
      #endif
      return ss.str();
    }

    string run_info() {
      stringstream ss;
      ss << "time    ";
      ss << get_date_string(std::chrono::system_clock::now()) << "\n";
      #if HAS_OMP
      ss << "threads " << omp_get_max_threads() << "\n";
      #endif
      return ss.str();
    }

    void initialize_config(
        const vector<string>& config_strings,
        const int config_options = jams::defaults::config_options) {
      using namespace libconfig;

      ::config.reset(new Config);
      ::config->setOptions(config_options);

      cout << "config files " << "\n";
      for (const auto& s : config_strings) {
        if (jams::system::file_exists(s)) {
          cout << "  " << s << "\n";
        }
      }

      jams::parse_config_strings(config_strings, ::config);

      std::string filename = jams::output::full_path_filename("combined.cfg");
      write_config(filename, ::config);
    }

    string choose_simulation_name(const jams::ProgramArgs &program_args) {
      string name = "jams";
      // specify a default name in case no other is found
      if (!program_args.simulation_name.empty()) {
        // name specified with command line flag
        name = trim(program_args.simulation_name);
      } else {
        // name after the first config file if one exists
        for (const auto& s : program_args.config_strings) {
          if (jams::system::file_exists(s)) {
            name = trim(file_basename_no_extension(s));
            break;
          }
        }
      }
      return name;
    }

    void initialize_simulation(const jams::ProgramArgs &program_args) {
      try {
        cout << jams::section("build info") << std::endl;
        cout << jams::build_info();
        cout << jams::section("run info") << std::endl;
        cout << jams::run_info();

        if (!program_args.output_path.empty()) {
          jams::instance().set_output_dir(program_args.output_path);
        }

        jams::Simulation simulation;

        ::simulation_name = choose_simulation_name(program_args);

        initialize_config(program_args.config_strings);

        jams::new_global_classes();

        jams::set_mode();

        if (jams::instance().mode() == Mode::GPU) {
          cout << "mode    GPU \n";
        } else {
          cout << "mode    CPU \n";
        }


        simulation.random_state = jams::instance().random_generator_internal_state();


        if (::config->exists("sim")) {
          simulation.verbose = jams::config_optional<bool>(config->lookup("sim"), "verbose", false);

          if (config->exists("sim.seed")) {
            simulation.random_seed = jams::config_required<unsigned long>(config->lookup("sim"), "seed");
            jams::instance().random_generator().seed(simulation.random_seed);
            cout << "seed    " << simulation.random_seed << "\n";
          }

          if (config->exists("sim.rng_state")) {
            auto state = jams::config_required<string>(config->lookup("sim"), "rng_state");
            istringstream(state) >> simulation.random_state;
          }
        }

        cout << "verbose " << simulation.verbose << "\n";
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
            solver->register_hamiltonian(
                Hamiltonian::create(hamiltonian_settings[i], globals::num_spins, solver->is_cuda_solver()));
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

        if (::config->exists("initializer")) {
          global_initializer(::config->lookup("initializer"));
        }
      }
      catch (const libconfig::SettingTypeException &stex) {
        jams_die("Config setting type error '%s'", stex.getPath());
      }
      catch (const libconfig::SettingNotFoundException &nfex) {
        jams_die("Required config setting not found '%s'", nfex.getPath());
      }
      catch (const jams::runtime_error &gex) {
        jams_die("%s", gex.what());
      }
      catch (std::exception &e) {
        jams_die("%s", e.what());
      }
      catch (...) {
        jams_die("Caught an unknown exception");
      }
    }

    void run_simulation() {
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
      catch (const jams::runtime_error &gex) {
        jams_die("%s", gex.what());
      }
      catch (std::exception &e) {
        jams_die("%s", e.what());
      }
      catch (...) {
        jams_die("Caught an unknown exception");
      }
    }

    void cleanup_simulation() {
      jams::delete_global_classes();
    }

    void global_initializer(const libconfig::Setting &settings) {
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

}
