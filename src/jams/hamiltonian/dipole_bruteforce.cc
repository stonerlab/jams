#ifdef HAS_OMP
#include <omp.h>
#endif

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "dipole_bruteforce.h"

DipoleHamiltonianCpuBruteforce::~DipoleHamiltonianCpuBruteforce() {
}

DipoleHamiltonianCpuBruteforce::DipoleHamiltonianCpuBruteforce(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  supercell_matrix_ = lattice->get_supercell().matrix();

  frac_positions_.resize(globals::num_spins);

    for (auto i = 0; i < globals::num_spins; ++i) {
      frac_positions_[i] = lattice->get_supercell().inverse_matrix()*lattice->atom_position(i);
    }

    std::vector<std::pair<Vec3f, int>> positions;

    Vec3i L_max = {1, 1, 1};
    for (auto i = 0; i < globals::num_spins; ++i) {
      // loop over periodic images of the simulation lattice
      // this means r_cutoff can be larger than the simulation cell
      Mat3 dipole_tensor = kZeroMat3;

      for (auto Lx = -L_max[0]; Lx < L_max[0] + 1; ++Lx) {
        for (auto Ly = -L_max[1]; Ly < L_max[1] + 1; ++Ly) {
          for (auto Lz = -L_max[2]; Lz < L_max[2] + 1; ++Lz) {
            Vec3i image_vector = {Lx, Ly, Lz};

//            auto r = lattice->generate_image_position(lattice->atom_position(i), image_vector);

            Vec3 frac_pos = lattice->cartesian_to_fractional(lattice->atom_position(i));

            bool skip = false;
            for (int n = 0; n < 3; ++n) {
              if (lattice->is_periodic(n)) {
                frac_pos[n] = frac_pos[n] + image_vector[n] * lattice->size()[n];
                if (frac_pos[n] < -lattice->size()[n]/2.0 || frac_pos[n] > 1.5*lattice->size()[n]) {
                  skip = true;
                  break;
                }
              }
            }

            if (skip) {
              continue;
            }
            auto r = lattice->fractional_to_cartesian(frac_pos);



            positions.emplace_back(std::make_pair(Vec3f{float(r[0]), float(r[1]), float(r[2])},i));
          }
        }
      }
    }

  auto distance_metric = [](const std::pair<Vec3f, int>& a, const std::pair<Vec3f, int>& b)->float {
      return norm_sq(a.first-b.first);
  };

  near_tree_ = new NearTree<std::pair<Vec3f, int>, NeartreeFunctorType>(distance_metric, positions);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_total_energy() {
    double e_total = 0.0;

       for (auto i = 0; i < globals::num_spins; ++i) {
           e_total += calculate_one_spin_energy(i);
       }

    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -0.5 * (s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy(const int i) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianCpuBruteforce::calculate_energies() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_one_spin_energy(i);
    }
}


__attribute__((hot))
void DipoleHamiltonianCpuBruteforce::calculate_one_spin_field(const int i, double h[3])
{
  using namespace globals;

  const auto w0 = kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));

  double hx = 0, hy = 0, hz = 0;

  std::vector<std::pair<Vec3f, int>> neighbours;
  const Vec3f r_i = {
      float(lattice->atom_position(i)[0]),
      float(lattice->atom_position(i)[1]),
      float(lattice->atom_position(i)[2])
  };
  near_tree_->find_in_radius(r_cutoff_, neighbours, {r_i, i});

  #if HAS_OMP
  #pragma omp parallel for reduction(+:hx, hy, hz)
  #endif
  for (auto n = 0; n < neighbours.size(); ++n) {
    const int j = neighbours[n].second;
    if (j == i) continue;

    const Vec3f r_ij =  neighbours[n].first - r_i;

    const auto r_abs_sq = norm_sq(r_ij);

    const auto sj_dot_r = s(j, 0) * r_ij[0] + s(j, 1) * r_ij[1] + s(j, 2) * r_ij[2];

    hx += w0 * mus(i) * mus(j) * (3.0 * r_ij[0] * sj_dot_r - r_abs_sq * s(j, 0)) / pow(r_abs_sq, 2.5);
    hy += w0 * mus(i) * mus(j) * (3.0 * r_ij[1] * sj_dot_r - r_abs_sq * s(j, 1)) / pow(r_abs_sq, 2.5);
    hz += w0 * mus(i) * mus(j) * (3.0 * r_ij[2] * sj_dot_r - r_abs_sq * s(j, 2)) / pow(r_abs_sq, 2.5);
  }

  h[0] = hx; h[1] = hy; h[2] = hz;
}

// --------------------------------------------------------------------------

void DipoleHamiltonianCpuBruteforce::calculate_fields() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        double h[3];

        calculate_one_spin_field(i, h);

        for (auto n : {0,1,2}) {
            field_(i, n) = h[n];
        }
    }
}