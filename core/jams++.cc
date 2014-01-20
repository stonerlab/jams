// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#define JAMS_VERSION "0.8.0"
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

#include "monitors/boltzmann.h"
#include "monitors/boltzmann_mag.h"
#include "monitors/energy.h"
#include "monitors/magnetisation.h"

#ifdef CUDA
#include <cublas.h>
#endif

namespace {
  Solver *solver;
  Physics *physics_package;
  double dt = 0.0;
  int steps_eq = 0;
  int steps_run = 0;
  int steps_out = 0;
  int  steps_vis = 0;
  int  steps_bin = 0;
  int steps_conv = 0;

  std::string convName;
  double convergence_tolerance_mean = 0.0;
  double convergence_tolerance_stddev = 0.0;

  bool energy_output_is_set = false;

  bool visual_output_is_set = false;
  bool binary_output_is_set = false;
  bool coarse_output_is_set = false;

  bool save_state_is_set = false;
  bool read_state_is_set = false;


  std::vector<Monitor*> monitor_list;
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
    }
    catch(const libconfig::FileIOException &fioex) {
      jams_error("I/O error while reading '%s'", config_filename.c_str());
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


    double init_temperature = 0.0;

    try {
      verbose_output_is_set = false;
      if (config.exists("sim.verbose_output")) {
        config.lookupValue("sim.verbose_output", verbose_output_is_set);
      }

      dt = config.lookup("sim.t_step");
      output.write("  * Timestep:           %1.8e\n", dt);


      double time_value = config.lookup("sim.t_eq");
      steps_eq = static_cast<int>(time_value/dt);
      output.write("  * Equilibration time: %1.8e (%lu steps)\n",
        time_value, steps_eq);

      time_value = config.lookup("sim.t_run");
      steps_run = static_cast<int>(time_value/dt);
      output.write("  * Run time:           %1.8e (%lu steps)\n",
        time_value, steps_run);

      time_value = config.lookup("sim.t_out");
      steps_out = static_cast<int>(time_value/dt);
      output.write("  * Output time:        %1.8e (%lu steps)\n",
        time_value, steps_out);

      if (config.exists("sim.convergence")) {
        config.lookupValue("sim.convergence", convName);
        config.lookupValue("sim.meanTolerance", convergence_tolerance_mean);
        config.lookupValue("sim.devTolerance", convergence_tolerance_stddev);

        time_value = config.lookup("sim.t_conv");
        steps_conv = static_cast<int>(time_value/dt);
        output.write("  * Convergence test time:        %1.8e (%lu steps)\n",
          time_value, steps_conv);
      }

      globals::h_app[0] = config.lookup("sim.h_app.[0]");
      globals::h_app[1] = config.lookup("sim.h_app.[1]");
      globals::h_app[2] = config.lookup("sim.h_app.[2]");

      if (config.exists("sim.read_state")) {
        read_state_is_set = true;
        output.write("  * Read state is ON\n");
      }

      if (config.exists("sim.save_state")) {
        config.lookupValue("sim.save_state", save_state_is_set);
        output.write("  * Save state is ON\n");
      }

      if (config.exists("sim.energy")) {
        config.lookupValue("sim.energy", energy_output_is_set);
        if (energy_output_is_set) {
          output.write("  * Energy calculation ON\n");
        } else {
          output.write("  * Energy calculation OFF\n");
        }
      }

      if (config.exists("sim.visualise")) {
        config.lookupValue("sim.visualise", visual_output_is_set);
        if (visual_output_is_set) {
          output.write("  * Visualisation is ON\n");
          time_value = config.lookup("sim.t_vis");
          steps_vis = static_cast<int>(time_value/dt);
          output.write("  * Visualisation time: %1.8e (%lu steps)\n",
            time_value, steps_vis);
        }
      } else {
        visual_output_is_set = false;
      }

      if (config.exists("sim.binary")) {
        config.lookupValue("sim.binary", binary_output_is_set);
        if (binary_output_is_set) {
          output.write("  * Binary output is ON\n");
          time_value = config.lookup("sim.t_bin");
          steps_bin = static_cast<int>(time_value/dt);
          output.write("  * Binary output time: %1.8e (%lu steps)\n",
            time_value, steps_bin);
        }
      } else {
        binary_output_is_set = false;
      }

      if (config.exists("lattice.coarse")) {
        coarse_output_is_set = true;
        if (coarse_output_is_set) {
          output.write("  * Coarse magnetisation map output is ON\n");
        }
      } else {
        coarse_output_is_set = false;
      }

      unsigned int randomseed;
      if (config.exists("sim.seed")) {
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

      lattice.create_from_config(config);

      if (binary_output_is_set) {
        std::ofstream binary_state_file
          (std::string(seedname+"_types.bin").c_str(),
          std::ios::binary | std::ios::out);

        lattice.output_spin_types_as_binary(binary_state_file);

        binary_state_file.close();
      }

      // If read_state is true then attempt to read state from binary
      // file. If this fails (num_spins != num_spins in the file) then JAMS
      // exits with an error to avoid mistakingly thinking the file was
      // loaded. NOTE: This must be done after lattice is created but
      // before the solver is initialized so the GPU solvers can copy the
      // correct spin array.

      if (read_state_is_set) {
        std::string binary_state_filename;

        config.lookupValue("sim.read_state", binary_state_filename);

        output.write("\nReading spin state from %s\n",
          binary_state_filename.c_str());

        std::ifstream binary_state_file(binary_state_filename.c_str(),
          std::ios::binary|std::ios::in);

        lattice.read_spin_state_from_binary(binary_state_file);

        binary_state_file.close();
      }

      if (config.exists("sim.solver")) {
        config.lookupValue("sim.solver", solname);
        solname = capitalize(solname);
      }

      output.write("\nInitialising physics module...\n");
      if (config.exists("sim.physics")) {
        config.lookupValue("sim.physics", physname);
        physname = capitalize(physname);

        if (physname == "FMR") {
          physics_package = Physics::Create(FMR);
        } else if (physname == "MFPT") {
          physics_package = Physics::Create(MFPT);
        } else if (physname == "TTM") {
          physics_package = Physics::Create(TTM);
        } else if (physname == "SPINWAVES") {
          physics_package = Physics::Create(SPINWAVES);
        } else if (physname == "DYNAMICSF") {
          physics_package = Physics::Create(DYNAMICSF);
        } else if (physname == "SQUARE") {
          physics_package = Physics::Create(SQUARE);
        } else if (physname == "FIELDCOOL") {
          physics_package = Physics::Create(FIELDCOOL);
        } else {
          jams_error("Unknown Physics package selected.");
        }

        libconfig::Setting &phys = config.lookup("physics");
        physics_package->initialize(phys);

      } else {
        physics_package = Physics::Create(EMPTY);
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

    solver->initialize(argc, argv, dt);
    solver->temperature(init_temperature);
  }

  // select monitors
  monitor_list.push_back(new MagnetisationMonitor());

  if (energy_output_is_set) {
    monitor_list.push_back(new EnergyMonitor());
  }

  if ( config.exists("sim.monitors") ) {
    libconfig::Setting &simcfg = config.lookup("sim");

    for (int i = 0; i < simcfg["monitors"].getLength(); ++i) {
      std::string monname;
      // monname = std::string(simcfg["monitors"][i].c_str());
      monname = capitalize(simcfg["monitors"][i].c_str());
      if (monname == "BOLTZMANN") {
        monitor_list.push_back(new BoltzmannMonitor());
      } else {
        jams_error("Unknown monitor selected.");
      }
    }
  }

  for (int i = 0; i < monitor_list.size(); ++i) {
    monitor_list[i]->initialize();
  }

  if (convName == "MAG") {
    output.write("Convergence for Magnetisation\n");
    monitor_list[0]->initialize_convergence(convMag, convergence_tolerance_mean,
      convergence_tolerance_stddev);
  } else if (convName == "PHI") {
    output.write("Convergence for Phi\n");
    monitor_list[0]->initialize_convergence(convPhi, convergence_tolerance_mean,
      convergence_tolerance_stddev);
  } else if (convName == "SINPHI") {
    output.write("Convergence for Sin(Phi)\n");
    monitor_list[0]->initialize_convergence(convSinPhi, convergence_tolerance_mean,
      convergence_tolerance_stddev);
  }
  output.write("StdDev Tolerance: %e\n", convergence_tolerance_mean,
    convergence_tolerance_stddev);
  return 0;
}

void jams_run() {
  using namespace globals;

  std::ofstream coarse_magnetisation_file;

  if (coarse_output_is_set) {
    coarse_magnetisation_file.open(std::string(seedname+"_map.dat").c_str());
  }


  globalSteps = 0;
  output.write("\n----Equilibration----\n");
  output.write("Running solver\n");
  for (int i = 0; i < steps_eq; ++i) {
    if (i%steps_out == 0) {
      solver->sync_device_data();

      monitor_list[0]->write(solver);
    }
    physics_package->run(solver->time(), dt);
    solver->temperature(globalTemperature);
    solver->run();
    globalSteps++;
  }

  output.write("\n----Data Run----\n");
  output.write("Running solver\n");
  std::clock_t start = std::clock();
  for (int i = 0; i < steps_run; ++i) {
    if (i%steps_out == 0) {
      solver->sync_device_data();
      for (int i = 0; i < monitor_list.size(); ++i) {
        monitor_list[i]->write(solver);
      }
      physics_package->monitor(solver->time(), dt);

      if (coarse_output_is_set) {
        lattice.output_coarse_magnetisation(coarse_magnetisation_file);
        coarse_magnetisation_file << "\n\n";
      }
    }
    if (visual_output_is_set) {
      if (i%steps_vis == 0) {
        int outcount = i/steps_vis;  // int divisible by modulo above

        std::ofstream vtu_state_file
          (std::string(seedname+"_"+zero_pad_number(outcount)+".vtu").c_str());

        lattice.output_spin_state_as_vtu(vtu_state_file);

        vtu_state_file.close();
      }
    }

    if (binary_output_is_set) {
      if (i%steps_bin == 0) {
        int outcount = i/steps_bin;  // int divisible by modulo above

        std::ofstream binary_state_file
          (std::string(seedname+"_"+zero_pad_number(outcount)+".bin").c_str(),
          std::ios::binary|std::ios::out);

        lattice.output_spin_state_as_binary(binary_state_file);

        binary_state_file.close();
      }
    }

    if (steps_conv > 0) {
      if ((i+1)%steps_conv == 0) {
        if (monitor_list[0]->has_converged() == true) {
          break;
        }
      }
    }

    physics_package->run(solver->time(), dt);
    solver->temperature(globalTemperature);
    solver->run();
    globalSteps++;
    for (int i = 0; i < monitor_list.size(); ++i) {
      monitor_list[i]->run();
    }
  }

  if (save_state_is_set) {
    output.write(
      "\n-------------------\nSaving spin state\n-------------------\n");

    std::ofstream binary_state_file
      (std::string(seedname+"_state.dat").c_str(),
      std::ios::out|std::ios::binary|std::ios::trunc);

    lattice.output_spin_state_as_binary(binary_state_file);

    binary_state_file.close();
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

  if (coarse_output_is_set) {
    coarse_magnetisation_file.close();
  }
}

void jams_finish() {
  if (solver != NULL) { delete solver; }
  if (physics_package != NULL) { delete physics_package; }
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

  output.write("\n****************************************\n");
  output.write("               JAMS ERROR               \n");
  output.write("****************************************\n");
  output.write("%s\n", buffer);

  jams_finish();
  exit(EXIT_FAILURE);
}
