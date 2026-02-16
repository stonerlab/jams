// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#include "version.h"

#include <fstream>
#include <memory>
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
#include "jams/helpers/output.h"
#include "jams/interface/config.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/timer.h"
#include "jams/helpers/progress_bar.h"

#include <jams/initializer/init_dispatcher.h>

namespace jams {

    void new_global_classes() {
      globals::lattice = new Lattice();
    }

    void delete_global_classes() {
      delete globals::solver;
      delete globals::lattice;
    }

    // Reads a vector of strings in order, combining to produce a config.
    //
    // If the string is an existent file name it is loaded as a config,
    // otherwise it is directly interpreted as a config string.
    void parse_config_strings(const std::vector<ConfigInput>& config_inputs, std::unique_ptr<libconfig::Config>& combined_config) {
      if (!combined_config) {
        combined_config = std::make_unique<libconfig::Config>();
      }

      for (const auto &input : config_inputs) {
        const std::string &s = input.value;
        libconfig::Config patch;
        if (!input.force_string && jams::system::file_exists(s)) {
          try {
            patch.readFile(s.c_str());
          }
          catch (libconfig::FileIOException &fex) {
            throw std::runtime_error("IO error opening config file: " + s);
          }
          catch (const libconfig::ParseException &pex) {
            throw std::runtime_error("Error parsing config file: "
                                     + std::string(pex.getFile()) + ":"
                                     + std::to_string(pex.getLine()) + ":"
                                     + std::string(pex.getError()));
          }
        } else {
          try {
            patch.readString(s.c_str());
          }
          catch (const libconfig::ParseException &pex) {
            std::stringstream ss;
            if (input.force_string) {
              ss << "Error parsing config string:\n";
            } else {
              ss << "File not found or error parsing config string:\n";
            }
            ss << "  '" << s << "'\n";
            ss << "line " << std::to_string(pex.getLine()) << ": " << std::string(pex.getError());

            throw std::runtime_error(ss.str());
          }
        }

        overwrite_config_settings(combined_config->getRoot(), patch.getRoot());
      }
    }

    void write_config(const std::string& filename, const std::unique_ptr<libconfig::Config> &cfg) {
      cfg->setFloatPrecision(jams::defaults::config_float_precision);
      cfg->writeFile(filename.c_str());
    }

    void set_mode() {
      std::string solver_name = globals::config->lookup("solver.module");
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

std::string build_info() {
  std::stringstream ss;
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
      ss << "  spglib     " << jams::build::spglib_version() << "\n";
      ss << "    " << find_and_replace(jams::build::spglib_libraries, ";", "\n    ") << "\n";
      ss << "  pcg        " << jams::build::pcg_version << "\n";
      ss << "    " << find_and_replace(jams::build::pcg_libraries, ";", "\n    ") << "\n";
      ss << "  hdf5       " << jams::build::hdf5_version << "\n";
      ss << "    include directories: \n";
      ss << "      " << find_and_replace(jams::build::hdf5_include_directories, ";", "\n      ") << "\n";
      ss << "    link libraries: \n";
      ss << "      " << find_and_replace(jams::build::hdf5_link_libraries, ";", "\n      ") << "\n";
      ss << "  fftw3      " << jams::build::fftw3_vendor << "\n";
      ss << "    include directories: \n";
      ss << "      " << find_and_replace(jams::build::fftw3_include_directories, ";", "\n      ") << "\n";
      ss << "    link libraries: \n";
      ss << "      " << find_and_replace(jams::build::fftw3_link_libraries, ";", "\n      ") << "\n";
      ss << "  cblas      " << jams::build::cblas_vendor << "\n";
      ss << "    include directories: \n";
      ss << "      " << find_and_replace(jams::build::cblas_include_directories, ";", "\n      ") << "\n";
      ss << "    link libraries: \n";
      ss << "      " << find_and_replace(jams::build::cblas_link_libraries, ";", "\n      ") << "\n";
      #if HAS_MKL
      ss << "  mkl        " << jams::build::mkl_version() << "\n";
      ss << "    link libraries: \n";
      ss << "      " << find_and_replace(jams::build::mkl_link_libraries, ";", "\n      ") << "\n";
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

std::string run_info() {
  std::stringstream ss;
      ss << "time    ";
      ss << get_date_string(std::chrono::system_clock::now()) << "\n";
      #if HAS_OMP
      ss << "threads " << omp_get_max_threads() << "\n";
      #endif
      return ss.str();
    }

    void initialize_config(
        const std::vector<ConfigInput>& config_inputs,
        const int config_options = jams::defaults::config_options) {

      ::globals::config = std::make_unique<libconfig::Config>();
      ::globals::config->setOptions(config_options);

      std::cout << "config files " << "\n";
      for (const auto& input : config_inputs) {
        if (!input.force_string && jams::system::file_exists(input.value)) {
          std::cout << "  " << input.value << "\n";
        }
      }

      jams::parse_config_strings(config_inputs, ::globals::config);

      std::string filename = jams::output::full_path_filename("combined.cfg");
      write_config(filename, ::globals::config);
    }

std::string choose_simulation_name(const jams::ProgramArgs &program_args) {
  std::string name = "jams";
      // specify a default name in case no other is found
      if (!program_args.simulation_name.empty()) {
        // name specified with command line flag
        name = trim(program_args.simulation_name);
      } else {
        // name after the first config file if one exists
      for (const auto& input : program_args.config_inputs) {
        if (!input.force_string && jams::system::file_exists(input.value)) {
          name = trim(file_basename_no_extension(input.value));
          break;
        }
      }
      }
      return name;
    }

    void initialize_simulation(const jams::ProgramArgs &program_args) {
      try {
        std::cout << "\nJAMS++ " << semantic_version(jams::build::description) << std::endl;

        if (contains(jams::build::description, "dirty")) {
          jams_warning("There are uncommitted changes in your git repository. DO NOT USE THIS BINARY FOR PRODUCTION CALCULATIONS.");
        }

        if (contains(jams::build::description, "unknown")) {
          jams_warning("JAMS version is unknown. DO NOT USE THIS BINARY FOR PRODUCTION CALCULATIONS.");
        }

        std::cout << jams::section("build info") << std::endl;
        std::cout << jams::build_info();
        std::cout << jams::section("run info") << std::endl;
        std::cout << jams::run_info();

        if (!program_args.output_path.empty()) {
          jams::instance().set_output_dir(program_args.output_path);
        }
        jams::instance().set_temp_directory_path(program_args.temp_directory_path);

        jams::Simulation simulation;

        ::globals::simulation_name = choose_simulation_name(program_args);

        initialize_config(program_args.config_inputs);

        jams::new_global_classes();

        jams::set_mode();

        if (jams::instance().mode() == Mode::GPU) {
          std::cout << "mode    GPU \n";
        } else {
          std::cout << "mode    CPU \n";
        }


        simulation.random_state = jams::instance().random_generator_internal_state();


        if (::globals::config->exists("sim")) {
          simulation.verbose = jams::config_optional<bool>(
              globals::config->lookup("sim"), "verbose", false);

          if (globals::config->exists("sim.seed")) {
            simulation.random_seed = jams::config_required<unsigned long>(
                globals::config->lookup("sim"), "seed");
            jams::instance().random_generator().seed(simulation.random_seed);
            std::cout << "seed    " << simulation.random_seed << "\n";
          }

          if (globals::config->exists("sim.rng_state")) {
            auto state = jams::config_required<std::string>(
                globals::config->lookup("sim"), "rng_state");
            std::istringstream(state) >> simulation.random_state;
          }
        }

        std::cout << "verbose " << simulation.verbose << "\n";
        std::cout << "rng state " << simulation.random_state << "\n";

        std::cout << jams::section("init lattice") << std::endl;

        globals::lattice->init_from_config(*::globals::config);

        std::cout << jams::section("init solver") << std::endl;

        globals::solver = Solver::create(globals::config->lookup("solver"));

        globals::solver->register_physics_module(Physics::create(
            globals::config->lookup("physics")));     // todo: fix this memory leak

        std::cout << jams::section("init hamiltonians") << std::endl;

        if (!::globals::config->exists("hamiltonians")) {
          throw jams::ConfigException(globals::config->getRoot(), "No hamiltonians group in config");
        } else {
          const libconfig::Setting &hamiltonian_settings = ::globals::config->lookup("hamiltonians");
          for (auto i = 0; i < hamiltonian_settings.getLength(); ++i) {
            globals::solver->register_hamiltonian(
                Hamiltonian::create(hamiltonian_settings[i], globals::num_spins, globals::solver->is_cuda_solver()));
          }
        }

        std::cout << jams::section("init monitors") << std::endl;

        if (!::globals::config->exists("monitors")) {
          jams_warning("No monitors in config");
        } else {
          const libconfig::Setting &monitor_settings = ::globals::config->lookup("monitors");
          for (auto i = 0; i < monitor_settings.getLength(); ++i) {
            globals::solver->register_monitor(Monitor::create(monitor_settings[i]));
          }
        }

        if (::globals::config->exists("initializer")) {
          jams::InitializerDispatcher::execute(::globals::config->lookup("initializer"));
        }
      }
      catch (const libconfig::SettingTypeException &stex) {
        jams::die("CONFIG ERROR", stex.getPath(), ": setting type error");
      }
      catch (const libconfig::SettingNotFoundException &nfex) {
        jams::die("CONFIG ERROR", nfex.getPath(), ": required setting not found");
      }
      catch (const jams::runtime_error &e) {
        jams::die("RUNTIME ERROR", e.what());
      }
      catch (const jams::ConfigException &e) {
        jams::die("CONFIG ERROR", e.what());
      }
      catch (std::exception &e) {
        jams::die("ERROR", e.what());
      }
      catch (...) {
        jams::die("UNKNOWN ERROR");
      }
    }

    void run_simulation() {
      try {
        std::cout << jams::section("running solver") << std::endl;
        std::cout << "start   " << get_date_string(std::chrono::system_clock::now()) << "\n" << std::endl;
        {
          ProgressBar progress;
          Timer<> timer;
          while (::globals::solver->is_running()) {
            ::globals::solver->update_physics_module();
            ::globals::solver->notify_monitors();

            if (::globals::solver->convergence_status() == Monitor::ConvergenceStatus::kConverged) {
              break;
            }

            ::globals::solver->run();

            progress.set(double(::globals::solver->iteration()) / double(::globals::solver->max_steps()));
            if (::globals::solver->iteration() % 1000 == 0) {
              std::cout << progress;
            }
          }
          std::cout << "\n\n";
          std::cout << "runtime " << timer.elapsed_time() << " seconds" << std::endl;

          std::cout << "finish  " << get_date_string(std::chrono::system_clock::now()) << "\n\n";
        }

        {
          std::cout << jams::section("running post process") << std::endl;
          std::cout << "start   " << get_date_string(std::chrono::system_clock::now()) << "\n" << std::endl;

          Timer<> timer;

          for (const auto& m : globals::solver->monitors()) {
            m->post_process();
          }
          std::cout << "runtime " << timer.elapsed_time() << " seconds" << std::endl;
          std::cout << "finish  " << get_date_string(std::chrono::system_clock::now()) << "\n\n";
        }

      }
      catch (const jams::runtime_error &gex) {
        jams::die("%s", gex.what());
      }
      catch (std::exception &e) {
        jams::die("%s", e.what());
      }
      catch (...) {
        jams::die("Caught an unknown exception");
      }
    }

    void cleanup_simulation() {
      jams::delete_global_classes();
    }

}
