// initializer_factory.cc                                                          -*-C++-*-
#include <jams/initializer/init_dispatcher.h>
#include <jams/initializer/init_bloch_domain_wall.h>
#include <jams/initializer/init_h5.h>

#include <stdexcept>
#include <jams/helpers/error.h>

#define DEFINED_INITIALIZER(module, type, settings) \
{ \
  if (lowercase(settings["module"]) == module) { \
    type::execute(settings);                      \
    return;                                                \
  } \
}

void jams::InitializerDispatcher::execute(const libconfig::Setting &settings) {
  // backwards compatibility for before we had named initializers the
  // "initializer" config section simply loaded data from a H5 file
  if (!settings.exists("module")) {
    jams_warning("No named initializer found. Using old behaviour to load from H5.");
    InitH5::execute(settings);
    return;
  }

  DEFINED_INITIALIZER("h5", InitH5, settings);
  DEFINED_INITIALIZER("domain_wall", InitDomainWall, settings);

  throw std::runtime_error("unknown initializer: " + std::string(settings["module"].c_str()));
}

#undef DEFINED_INITIALIZER