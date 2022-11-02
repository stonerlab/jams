// init_h5.cc                                                          -*-C++-*-
#include <jams/initializer/init_h5.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/helpers/load.h>

#include <string>


void jams::InitH5::execute(const libconfig::Setting &settings) {
  if (settings.exists("spins")) {
    std::string file_name = settings["spins"];
    std::cout << "reading spin data from file " << file_name << "\n";
    load_array_from_file(file_name, "/spins", globals::s);
  }

  if (settings.exists("alpha")) {
    std::string file_name = settings["alpha"];
    std::cout << "reading alpha data from file " << file_name << "\n";
    load_array_from_file(file_name, "/alpha", globals::alpha);
  }

  if (settings.exists("mus")) {
    std::string file_name = settings["mus"];
    std::cout << "reading mus data from file " << file_name << "\n";
    load_array_from_file(file_name, "/mus", globals::mus);
  }

  if (settings.exists("gyro")) {
    std::string file_name = settings["gyro"];
    std::cout << "reading gyro data from file " << file_name << "\n";
    load_array_from_file(file_name, "/gyro", globals::gyro);
  }
}
