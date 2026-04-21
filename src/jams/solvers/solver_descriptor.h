#ifndef JAMS_SOLVERS_SOLVER_DESCRIPTOR_H
#define JAMS_SOLVERS_SOLVER_DESCRIPTOR_H

#include <libconfig.h++>

#include <stdexcept>
#include <string>

#include "jams/helpers/utils.h"

namespace jams::solvers {

enum class Backend {
  CPU,
  GPU,
};

enum class IntegratorKind {
  Unknown,
  Null,
  Rotations,
  Heun,
  RK4,
  RKMK2,
  RKMK4,
  SemiImplicit,
  DM,
  MonteCarloMetropolis,
  MonteCarloConstrained,
  MonteCarloMetadynamics,
};

enum class EquationKind {
  None,
  LLG,
  GSE,
  LLLorentzian,
};

struct SolverDescriptor {
  Backend backend = Backend::CPU;
  IntegratorKind integrator = IntegratorKind::Unknown;
  EquationKind equation = EquationKind::None;
  std::string legacy_module;
  bool legacy_sot_alias = false;
};

inline std::string backend_name(const Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return "cpu";
    case Backend::GPU:
      return "gpu";
  }
  throw std::runtime_error("unknown solver backend enum");
}

inline std::string canonical_solver_name(const SolverDescriptor& descriptor) {
  if (!descriptor.legacy_module.empty()) {
    return descriptor.legacy_module;
  }

  switch (descriptor.integrator) {
    case IntegratorKind::Null:
      return "null";
    case IntegratorKind::Rotations:
      return "rotations-" + backend_name(descriptor.backend);
    case IntegratorKind::Heun:
      return "llg-heun-" + backend_name(descriptor.backend);
    case IntegratorKind::RK4:
      if (descriptor.equation == EquationKind::GSE) {
        return "gse-rk4-" + backend_name(descriptor.backend);
      }
      if (descriptor.equation == EquationKind::LLLorentzian) {
        return "ll-lorentzian-rk4-" + backend_name(descriptor.backend);
      }
      return "llg-rk4-" + backend_name(descriptor.backend);
    case IntegratorKind::RKMK2:
      return "llg-rkmk2-" + backend_name(descriptor.backend);
    case IntegratorKind::RKMK4:
      return "llg-rkmk4-" + backend_name(descriptor.backend);
    case IntegratorKind::SemiImplicit:
      return "llg-simp-" + backend_name(descriptor.backend);
    case IntegratorKind::DM:
      return "llg-dm-" + backend_name(descriptor.backend);
    case IntegratorKind::MonteCarloMetropolis:
      return "monte-carlo-metropolis-" + backend_name(descriptor.backend);
    case IntegratorKind::MonteCarloConstrained:
      return "monte-carlo-constrained-" + backend_name(descriptor.backend);
    case IntegratorKind::MonteCarloMetadynamics:
      return "monte-carlo-metadynamics-" + backend_name(descriptor.backend);
    case IntegratorKind::Unknown:
      break;
  }

  throw std::runtime_error("cannot infer canonical solver name");
}

inline Backend parse_backend_name(const std::string& backend_name) {
  const auto lowered = lowercase(backend_name);
  if (lowered == "cpu") {
    return Backend::CPU;
  }
  if (lowered == "gpu") {
    return Backend::GPU;
  }
  throw std::runtime_error("unknown solver backend " + backend_name);
}

inline IntegratorKind parse_integrator_name(const std::string& integrator_name) {
  const auto lowered = lowercase(integrator_name);
  if (lowered == "null") {
    return IntegratorKind::Null;
  }
  if (lowered == "rotations") {
    return IntegratorKind::Rotations;
  }
  if (lowered == "heun") {
    return IntegratorKind::Heun;
  }
  if (lowered == "rk4") {
    return IntegratorKind::RK4;
  }
  if (lowered == "rkmk2") {
    return IntegratorKind::RKMK2;
  }
  if (lowered == "rkmk4") {
    return IntegratorKind::RKMK4;
  }
  if (lowered == "simp" || lowered == "semi-implicit" || lowered == "semi_implicit") {
    return IntegratorKind::SemiImplicit;
  }
  if (lowered == "dm") {
    return IntegratorKind::DM;
  }
  if (lowered == "monte-carlo-metropolis") {
    return IntegratorKind::MonteCarloMetropolis;
  }
  if (lowered == "monte-carlo-constrained") {
    return IntegratorKind::MonteCarloConstrained;
  }
  if (lowered == "monte-carlo-metadynamics") {
    return IntegratorKind::MonteCarloMetadynamics;
  }
  throw std::runtime_error("unknown solver integrator " + integrator_name);
}

inline EquationKind parse_equation_name(const std::string& equation_name) {
  const auto lowered = lowercase(equation_name);
  if (lowered == "llg") {
    return EquationKind::LLG;
  }
  if (lowered == "gse") {
    return EquationKind::GSE;
  }
  if (lowered == "ll-lorentzian" || lowered == "ll_lorentzian") {
    return EquationKind::LLLorentzian;
  }
  throw std::runtime_error("unknown dynamics equation " + equation_name);
}

inline SolverDescriptor describe_legacy_solver_module(const std::string& module_name) {
  const auto lowered = lowercase(module_name);

  if (lowered == "null") {
    return {Backend::CPU, IntegratorKind::Null, EquationKind::None, lowered, false};
  }
  if (lowered == "rotations-cpu") {
    return {Backend::CPU, IntegratorKind::Rotations, EquationKind::None, lowered, false};
  }
  if (lowered == "monte-carlo-metropolis-cpu") {
    return {Backend::CPU, IntegratorKind::MonteCarloMetropolis, EquationKind::None, lowered, false};
  }
  if (lowered == "monte-carlo-constrained-cpu") {
    return {Backend::CPU, IntegratorKind::MonteCarloConstrained, EquationKind::None, lowered, false};
  }
  if (lowered == "monte-carlo-metadynamics-cpu") {
    return {Backend::CPU, IntegratorKind::MonteCarloMetadynamics, EquationKind::None, lowered, false};
  }
  if (lowered == "llg-heun-cpu") {
    return {Backend::CPU, IntegratorKind::Heun, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rk4-cpu") {
    return {Backend::CPU, IntegratorKind::RK4, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rkmk2-cpu") {
    return {Backend::CPU, IntegratorKind::RKMK2, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rkmk4-cpu") {
    return {Backend::CPU, IntegratorKind::RKMK4, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-simp-cpu") {
    return {Backend::CPU, IntegratorKind::SemiImplicit, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-dm-cpu") {
    return {Backend::CPU, IntegratorKind::DM, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-sot-rk4-cpu") {
    return {Backend::CPU, IntegratorKind::RK4, EquationKind::LLG, lowered, true};
  }
  if (lowered == "gse-rk4-cpu") {
    return {Backend::CPU, IntegratorKind::RK4, EquationKind::GSE, lowered, false};
  }
  if (lowered == "ll-lorentzian-rk4-cpu") {
    return {Backend::CPU, IntegratorKind::RK4, EquationKind::LLLorentzian, lowered, false};
  }
  if (lowered == "llg-heun-gpu") {
    return {Backend::GPU, IntegratorKind::Heun, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rk4-gpu") {
    return {Backend::GPU, IntegratorKind::RK4, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rkmk2-gpu") {
    return {Backend::GPU, IntegratorKind::RKMK2, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-rkmk4-gpu") {
    return {Backend::GPU, IntegratorKind::RKMK4, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-simp-gpu") {
    return {Backend::GPU, IntegratorKind::SemiImplicit, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-dm-gpu") {
    return {Backend::GPU, IntegratorKind::DM, EquationKind::LLG, lowered, false};
  }
  if (lowered == "llg-sot-rk4-gpu") {
    return {Backend::GPU, IntegratorKind::RK4, EquationKind::LLG, lowered, true};
  }
  if (lowered == "gse-rk4-gpu") {
    return {Backend::GPU, IntegratorKind::RK4, EquationKind::GSE, lowered, false};
  }
  if (lowered == "ll-lorentzian-rk4-gpu") {
    return {Backend::GPU, IntegratorKind::RK4, EquationKind::LLLorentzian, lowered, false};
  }

  throw std::runtime_error("unknown solver " + module_name);
}

inline EquationKind infer_default_equation(const IntegratorKind integrator) {
  switch (integrator) {
    case IntegratorKind::Heun:
    case IntegratorKind::RKMK2:
    case IntegratorKind::RKMK4:
    case IntegratorKind::SemiImplicit:
    case IntegratorKind::DM:
      return EquationKind::LLG;
    case IntegratorKind::RK4:
      return EquationKind::LLG;
    default:
      return EquationKind::None;
  }
}

inline SolverDescriptor describe_solver_setting(const libconfig::Setting& solver_settings,
                                               const libconfig::Config& full_config) {
  if (solver_settings.exists("module")) {
    return describe_legacy_solver_module(static_cast<const char*>(solver_settings["module"]));
  }

  SolverDescriptor descriptor;
  descriptor.backend = parse_backend_name(static_cast<const char*>(solver_settings["backend"]));
  descriptor.integrator = parse_integrator_name(static_cast<const char*>(solver_settings["integrator"]));

  if (full_config.exists("dynamics")) {
    const auto& dynamics_settings = full_config.lookup("dynamics");
    if (dynamics_settings.exists("equation")) {
      descriptor.equation = parse_equation_name(static_cast<const char*>(dynamics_settings["equation"]));
    } else {
      descriptor.equation = infer_default_equation(descriptor.integrator);
    }
  } else {
    descriptor.equation = infer_default_equation(descriptor.integrator);
  }

  return descriptor;
}

inline SolverDescriptor describe_solver_config(const libconfig::Config& full_config) {
  return describe_solver_setting(full_config.lookup("solver"), full_config);
}

inline bool solver_uses_gpu(const libconfig::Config& full_config) {
  return describe_solver_config(full_config).backend == Backend::GPU;
}

}  // namespace jams::solvers

#endif  // JAMS_SOLVERS_SOLVER_DESCRIPTOR_H
