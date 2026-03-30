// Copyright 2014 Joseph Barker. All rights reserved.

#define GLOBALORIGIN
#include "version.h"

#include <fstream>
#include <memory>
#include <cctype>
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

    namespace {
      enum class PathTokenKind {
        Name,
        Index,
        Append,
      };

      struct PathToken {
        PathTokenKind kind;
        std::string name;
        int index = -1;
      };

      struct PathAssignmentParseState {
        int brace_depth = 0;
        int bracket_depth = 0;
        int paren_depth = 0;
        bool in_string = false;
        bool escaped = false;
      };

      bool is_top_level(const PathAssignmentParseState& state) {
        return !state.in_string
            && state.brace_depth == 0
            && state.bracket_depth == 0
            && state.paren_depth == 0;
      }

      void update_parse_state(char c, PathAssignmentParseState& state) {
        if (state.in_string) {
          if (state.escaped) {
            state.escaped = false;
            return;
          }
          if (c == '\\') {
            state.escaped = true;
            return;
          }
          if (c == '"') {
            state.in_string = false;
          }
          return;
        }

        switch (c) {
          case '"':
            state.in_string = true;
            return;
          case '{':
            ++state.brace_depth;
            return;
          case '}':
            --state.brace_depth;
            return;
          case '[':
            ++state.bracket_depth;
            return;
          case ']':
            --state.bracket_depth;
            return;
          case '(':
            ++state.paren_depth;
            return;
          case ')':
            --state.paren_depth;
            return;
          default:
            return;
        }
      }

      std::vector<std::string> split_top_level_statements(const std::string& input) {
        std::vector<std::string> statements;
        std::string current;
        PathAssignmentParseState state;

        for (const char c : input) {
          if (c == ';' && is_top_level(state)) {
            if (!trim(current).empty()) {
              statements.push_back(trim(current));
            }
            current.clear();
            continue;
          }

          current += c;
          update_parse_state(c, state);
        }

        if (!trim(current).empty()) {
          statements.push_back(trim(current));
        }

        return statements;
      }

      size_t find_top_level_assignment(const std::string& statement) {
        PathAssignmentParseState state;

        for (size_t i = 0; i < statement.size(); ++i) {
          const char c = statement[i];
          if (c == '=' && is_top_level(state)) {
            return i;
          }
          update_parse_state(c, state);
        }

        return std::string::npos;
      }

      bool assign_scalar_setting(libconfig::Setting& destination, const libconfig::Setting& source) {
        if (source.getType() == libconfig::Setting::Type::TypeInt) {
          destination = static_cast<int>(source);
          destination.setFormat(source.getFormat());
          return true;
        }

        if (source.getType() == libconfig::Setting::Type::TypeInt64) {
          destination = static_cast<int64_t>(source);
          destination.setFormat(source.getFormat());
          return true;
        }

        if (source.getType() == libconfig::Setting::Type::TypeFloat) {
          destination = static_cast<double>(source);
          return true;
        }

        if (source.getType() == libconfig::Setting::Type::TypeString) {
          destination = source.c_str();
          return true;
        }

        if (source.getType() == libconfig::Setting::Type::TypeBoolean) {
          destination = static_cast<bool>(source);
          return true;
        }

        return false;
      }

      bool copy_setting_value(libconfig::Setting& destination, const libconfig::Setting& source) {
        if (source.isGroup()) {
          for (auto i = 0; i < source.getLength(); ++i) {
            auto& child = destination.add(source[i].getName(), source[i].getType());
            if (!copy_setting_value(child, source[i])) {
              return false;
            }
          }
          return true;
        }

        if (source.isList() || source.isArray()) {
          for (auto i = 0; i < source.getLength(); ++i) {
            auto& child = destination.add(source[i].getType());
            if (!copy_setting_value(child, source[i])) {
              return false;
            }
          }
          return true;
        }

        return assign_scalar_setting(destination, source);
      }

      bool assign_named_value(libconfig::Setting& parent, const std::string& name, const libconfig::Setting& value) {
        if (value.isGroup() || value.isList()) {
          if (parent.exists(name)) {
            auto& existing = parent.lookup(name);
            if (existing.getType() == value.getType()) {
              overwrite_config_settings(existing, value);
              return true;
            }
            parent.remove(name);
          }

          auto& created = parent.add(name, value.getType());
          return copy_setting_value(created, value);
        }

        if (parent.exists(name)) {
          parent.remove(name);
        }

        auto& created = parent.add(name, value.getType());
        return copy_setting_value(created, value);
      }

      bool assign_indexed_value(libconfig::Setting& parent, const int index, const libconfig::Setting& value) {
        if (parent.getLength() <= index) {
          return false;
        }

        auto& existing = parent[index];

        if (value.isGroup() || value.isList()) {
          if (existing.getType() != value.getType()) {
            return false;
          }
          overwrite_config_settings(existing, value);
          return true;
        }

        if (existing.getType() != value.getType()) {
          return false;
        }

        return assign_scalar_setting(existing, value);
      }

      bool append_list_value(libconfig::Setting& parent, const libconfig::Setting& value) {
        auto& created = parent.add(value.getType());
        return copy_setting_value(created, value);
      }

      libconfig::Setting::Type container_setting_type(const PathTokenKind token_kind) {
        if (token_kind == PathTokenKind::Name) {
          return libconfig::Setting::Type::TypeGroup;
        }
        return libconfig::Setting::Type::TypeList;
      }

      bool ensure_named_container(libconfig::Setting& parent,
                                  const std::string& name,
                                  const PathTokenKind next_token_kind,
                                  libconfig::Setting*& child) {
        const auto expected_type = container_setting_type(next_token_kind);
        if (!parent.exists(name)) {
          child = &parent.add(name, expected_type);
          return true;
        }

        child = &parent.lookup(name);
        if (next_token_kind == PathTokenKind::Name) {
          return child->isGroup();
        }
        if (next_token_kind == PathTokenKind::Append) {
          return child->isList();
        }
        return child->isList() || child->isArray();
      }

      bool ensure_indexed_container(libconfig::Setting& parent,
                                    const int index,
                                    const PathTokenKind next_token_kind,
                                    libconfig::Setting*& child) {
        if (parent.getLength() <= index) {
          return false;
        }

        child = &parent[index];
        if (next_token_kind == PathTokenKind::Name) {
          return child->isGroup();
        }
        if (next_token_kind == PathTokenKind::Append) {
          return child->isList();
        }
        return child->isList() || child->isArray();
      }

      bool parse_setting_path(const std::string& lhs, std::vector<PathToken>& tokens) {
        const auto path = trim(lhs);
        if (path.empty()) {
          return false;
        }

        auto i = size_t{0};
        const auto skip_whitespace = [&]() {
          while (i < path.size() && std::isspace(static_cast<unsigned char>(path[i]))) {
            ++i;
          }
        };

        const auto parse_name = [&](std::string& name) {
          skip_whitespace();
          if (i >= path.size()) {
            return false;
          }

          const auto start = i;
          if (!(std::isalpha(static_cast<unsigned char>(path[i])) || path[i] == '_')) {
            return false;
          }

          ++i;
          while (i < path.size()) {
            const auto ch = static_cast<unsigned char>(path[i]);
            if (!(std::isalnum(ch) || ch == '_')) {
              break;
            }
            ++i;
          }

          name = path.substr(start, i - start);
          return true;
        };

        std::string root_name;
        if (!parse_name(root_name)) {
          return false;
        }
        tokens.push_back({PathTokenKind::Name, root_name, -1});

        while (i < path.size()) {
          skip_whitespace();
          if (i >= path.size()) {
            break;
          }

          if (path[i] == '.') {
            ++i;
            std::string name;
            if (!parse_name(name)) {
              return false;
            }
            tokens.push_back({PathTokenKind::Name, name, -1});
            continue;
          }

          if (path[i] == '[') {
            ++i;
            skip_whitespace();

            if (i < path.size() && path[i] == ']') {
              ++i;
              tokens.push_back({PathTokenKind::Append, "", -1});
              continue;
            }

            const auto index_start = i;
            while (i < path.size() && std::isdigit(static_cast<unsigned char>(path[i]))) {
              ++i;
            }
            if (index_start == i) {
              return false;
            }

            const auto index = std::stoi(path.substr(index_start, i - index_start));
            skip_whitespace();
            if (i >= path.size() || path[i] != ']') {
              return false;
            }
            ++i;
            tokens.push_back({PathTokenKind::Index, "", index});
            continue;
          }

          return false;
        }

        return tokens.size() > 1;
      }

      bool apply_path_assignment(libconfig::Setting& current,
                                 const std::vector<PathToken>& tokens,
                                 const size_t token_index,
                                 const libconfig::Setting& value) {
        const auto& token = tokens[token_index];
        const auto is_last = (token_index + 1 == tokens.size());

        if (token.kind == PathTokenKind::Name) {
          if (!current.isGroup()) {
            return false;
          }

          if (is_last) {
            return assign_named_value(current, token.name, value);
          }

          libconfig::Setting* child = nullptr;
          if (!ensure_named_container(current, token.name, tokens[token_index + 1].kind, child)) {
            return false;
          }

          return apply_path_assignment(*child, tokens, token_index + 1, value);
        }

        if (token.kind == PathTokenKind::Index) {
          if (!current.isList() && !current.isArray()) {
            return false;
          }

          if (is_last) {
            return assign_indexed_value(current, token.index, value);
          }

          libconfig::Setting* child = nullptr;
          if (!ensure_indexed_container(current, token.index, tokens[token_index + 1].kind, child)) {
            return false;
          }

          return apply_path_assignment(*child, tokens, token_index + 1, value);
        }

        if (!current.isList() || !is_last) {
          return false;
        }

        return append_list_value(current, value);
      }

      bool try_apply_path_assignment_string(const std::string& input, libconfig::Setting& target_root) {
        const auto statements = split_top_level_statements(input);
        if (statements.empty()) {
          return false;
        }

        for (const auto& statement : statements) {
          const auto assignment_pos = find_top_level_assignment(statement);
          if (assignment_pos == std::string::npos) {
            return false;
          }

          const auto lhs = trim(statement.substr(0, assignment_pos));
          const auto rhs = trim(statement.substr(assignment_pos + 1));

          std::vector<PathToken> tokens;
          if (!parse_setting_path(lhs, tokens) || rhs.empty()) {
            return false;
          }

          for (auto i = size_t{0}; i + 1 < tokens.size(); ++i) {
            if (tokens[i].kind == PathTokenKind::Append) {
              throw std::runtime_error("Config path append must assign a whole list element:\n  '" + statement + "'");
            }
          }

          libconfig::Config value;
          try {
            value.readString(("value = " + rhs + ";").c_str());
          } catch (const libconfig::ParseException&) {
            return false;
          }

          if (!apply_path_assignment(target_root, tokens, 0, value.lookup("value"))) {
            throw std::runtime_error("Invalid config path assignment:\n  '" + statement + "'");
          }
        }

        return true;
      }

      std::string libconfig_string_literal(const std::string& value) {
        auto escaped = find_and_replace(value, "\\", "\\\\");
        escaped = find_and_replace(escaped, "\"", "\\\"");
        return "\"" + escaped + "\"";
      }

      std::vector<ConfigInput> config_inputs_with_cli_overrides(
          const ProgramArgs& program_args) {
        auto config_inputs = program_args.config_inputs;

        if (!program_args.initial_spin_filename.empty()) {
          config_inputs.push_back({
              "lattice.spins = "
                  + libconfig_string_literal(program_args.initial_spin_filename)
                  + ";",
              true});
        }

        return config_inputs;
      }
    }

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
            if (try_apply_path_assignment_string(s, combined_config->getRoot())) {
              continue;
            }
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

        initialize_config(config_inputs_with_cli_overrides(program_args));

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
