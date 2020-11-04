#pragma once

#include <string>

#include "jams/helpers/consts.h"

namespace jams {
    namespace testing {
        namespace dipole {
            const std::string config_basic_cpu(R"(
              solver : {
                module = "llg-heun-cpu";
                t_step = 1.0e-16;
                t_min  = 1.0e-16;
                t_max  = 1.0e-16;
              };

              physics : {
                temperature = 1.0;
              };
              )");

            const std::string config_basic_gpu(R"(
              solver : {
                module = "llg-heun-gpu";
                t_step = 1.0e-16;
                t_min  = 1.0e-16;
                t_max  = 1.0e-16;
              };

              physics : {
                temperature = 1.0;
              };
              )");

            const std::string config_unitcell_sc(R"(
              materials = (
                { name      = "Fe";
                  moment    = 2.0;
                  spin      = [1.0, 0.0, 0.0];
                }
              );

              unitcell : {
                parameter = 0.3e-9;

                basis = (
                  [ 1.0, 0.0, 0.0],
                  [ 0.0, 1.0, 0.0],
                  [ 0.0, 0.0, 1.0]);
                positions = (
                  ("Fe", [0.0, 0.0, 0.0])
                  );
              };
              )");

            const std::string config_unitcell_sc_2_atom(R"(
              materials = (
                { name      = "FeA";
                  moment    = 2.0;
                  spin      = [1.0, 0.0, 0.0];
                },
                { name      = "FeB";
                  moment    = 2.0;
                  spin      = [1.0, 0.0, 0.0];
                }
              );

              unitcell : {
                parameter = 0.3e-9;

                basis = (
                  [ 2.0, 0.0, 0.0],
                  [ 0.0, 1.0, 0.0],
                  [ 0.0, 0.0, 1.0]);
                positions = (
                  ("FeA", [0.0, 0.0, 0.0]),
                  ("FeB", [0.5, 0.0, 0.0])
                  );
              };
              )");

            const std::string config_unitcell_bcc_2_atom(R"(
              materials = (
                { name      = "FeA";
                  moment    = 2.0;
                  spin      = [1.0, 0.0, 0.0];
                },
                { name      = "FeB";
                  moment    = 1.0;
                  spin      = [1.0, 0.0, 0.0];
                }
              );

              unitcell : {
                parameter = 0.3e-9;

                basis = (
                  [ 1.0, 0.0, 0.0],
                  [ 0.0, 1.0, 0.0],
                  [ 0.0, 0.0, 1.0]);
                positions = (
                  ("FeA", [0.0, 0.0, 0.0]),
                  ("FeB", [0.5, 0.5, 0.0])
                  );
              };
              )");

            const std::string config_unitcell_sc_AFM(R"(
              materials = (
                { name      = "FeA";
                  moment    = 2.0;
                  spin      = [0.0, 0.0, 1.0];
                  transform = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
                },
                { name      = "FeB";
                  moment    = 2.0;
                  spin      = [0.0, 0.0, -1.0];
                  transform = ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]);
                }
              );

              unitcell : {
                parameter = 0.3e-9;

                basis = (
                  [ 2.0, 0.0, 0.0],
                  [ 0.0, 2.0, 0.0],
                  [ 0.0, 0.0, 1.0]);
                positions = (
                  ("FeA", [0.0, 0.0, 0.0]),
                  ("FeA", [0.5, 0.5, 0.0]),
                  ("FeB", [0.5, 0.0, 0.0]),
                  ("FeB", [0.0, 0.5, 0.0])
                  );
              };
              )");

            const std::string config_cubic_unitcell(R"(
              materials = (
                { name      = "FM";
                  moment    = 1.65;
                  alpha     = 0.01;
                  spin      = [0.0, 0.0, 1.0];
                }
              );

              unitcell: {
                parameter = 5.90298e-10
                basis = (
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]);
                positions = (
                  ("FM", [0.0, 0.0, 0.0]) );
               };
               )");

            const std::string config_non_cubic_unitcell(R"(
              materials = (
                { name      = "FM";
                  moment    = 1.65;
                  alpha     = 0.01;
                  spin      = [0.0, 0.0, 1.0];
                }
              );

              unitcell: {
                parameter = 5.90298e-10
                basis = (
                  [0.5, -0.25, 0.0],
                  [0.0, 0.4330127019, 0.2886751346],
                  [0.0, 0.0, 0.4082482905]);
                positions = (
                  ("FM", [0.0, 0.0, 0.0]) );
               };
               )");

            std::string config_lattice(const Vec3 &size, const Vec3b &pbc) {
              std::stringstream ss;
              ss << "lattice : {\n";
              ss << "size = [" << size[0] << ", " << size[1] << ", " << size[2] << "];\n";
              ss << std::boolalpha << "periodic  = [" << pbc[0] << ", " << pbc[1] << ", " << pbc[2] << "];\n";
              ss << "};\n";
              return ss.str();
            }

            std::string config_dipole(const std::string &name, const double &r_cutoff) {
              std::stringstream ss;
              ss << "hamiltonians = ({\n";
              ss << "module = \"" << name << "\";\n";
              ss << "r_cutoff = " << std::to_string(r_cutoff) << ";\n";
              ss << "});\n";
              return ss.str();
            }
        }
    }
}