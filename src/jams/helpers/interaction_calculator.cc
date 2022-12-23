//
// Created by Joseph Barker on 2019-11-13.
//

#include "jams/core/globals.h"
#include "jams/helpers/interaction_calculator.h"

#include <iostream>

namespace jams {
    struct InteractionShell {
        int index = 0;
        double radius = 0.0;
        std::vector<InteractionData> interactions;
    };

    void interaction_calculator(const Cell &unitcell, const std::vector<Atom> &motif, const double r_max,
                                       const double eps) {
      // assumes motif is in fractional coordinates
      Vec3i num_cells = {
          int(ceil(r_max / norm(unitcell.a()))),
          int(ceil(r_max / norm(unitcell.b()))),
          int(ceil(r_max / norm(unitcell.c())))
      };

      std::vector<Atom> atoms;
      // generate all atoms up to r_max
      int counter = 0;
      for (auto i = -num_cells[0]; i < num_cells[0] + 1; ++i) {
        for (auto j = -num_cells[1]; j < num_cells[1] + 1; ++j) {
          for (auto k = -num_cells[2]; k < num_cells[2] + 1; ++k) {
            auto cell_offset = Vec3i{{i, j, k}};
            for (auto n = 0; n < motif.size(); ++n) {
              auto r = motif[n].position + cell_offset;
              atoms.push_back({counter, motif[n].material_index, n, r});
              counter++;
            }
          }
        }
      }

      for (auto n = 0; n < motif.size(); ++n) {
        auto origin = motif[n].position;

        auto cartesian_distance = [unitcell, origin](const Atom &a) -> double {
            return norm(unitcell.matrix() * (a.position - origin));
        };

        auto cartesian_distance_comp = [&](const Atom &a, const Atom &b) -> bool {
            return cartesian_distance(a) < cartesian_distance(b);
        };

        sort(atoms.begin(), atoms.end(), cartesian_distance_comp);

        std::ostringstream ss;


        std::vector<InteractionShell> interaction_shells;

        InteractionShell shell;

        for (const auto &atom : atoms) {
          const auto radius = cartesian_distance(atom);

          if (!approximately_equal(radius, shell.radius, eps)) {
            if (!shell.interactions.empty()) {
              interaction_shells.push_back(shell);
            }
            // start a new shell
            shell.index++;
            shell.radius = radius;
            shell.interactions.clear();
          }

          if (approximately_zero(radius, jams::defaults::lattice_tolerance)) {  // self interaction
            continue;
          }

          if (radius > r_max) {  // we can break not continue because the atom array is sorted
            break;
          }

          InteractionData data;

          data.unit_cell_pos_i = n;
          data.unit_cell_pos_j = atom.motif_index;
          data.r_ij = unitcell.matrix() * (atom.position - origin);
          data.type_i = ::globals::lattice->material_name(motif[n].material_index);
          data.type_j = ::globals::lattice->material_name(atom.material_index);

          shell.interactions.push_back(data);

        }

        std::cout
            << "# pos i |  pos j | type i | type j |      Rij_x |      Rij_y |      Rij_z |      Rij_u |      Rij_v |      Rij_w\n";
        for (const auto &s : interaction_shells) {
          std::cout << "# shell: " << s.index << " interactions: " << s.interactions.size() << " radius: " << s.radius
               << std::endl;
          for (const auto &interaction : s.interactions) {
            std::cout << std::setw(8) << jams::fmt::integer << interaction.unit_cell_pos_i + 1 << " ";
            std::cout << std::setw(8) << jams::fmt::integer << interaction.unit_cell_pos_j + 1 << " ";
            std::cout << std::setw(8) << interaction.type_i << " ";
            std::cout << std::setw(8) << interaction.type_j << " ";
            std::cout << jams::fmt::decimal << interaction.r_ij << " ";
            std::cout << jams::fmt::decimal << unitcell.inverse_matrix() * interaction.r_ij << std::endl;
          }
        }
        std::cout << '#';
        std::cout << std::string(79, '-');
        std::cout << std::endl;

      }
    }
}