#include <set>
#include <fstream>

#include "jams/helpers/exception.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/core/interactions.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"

#include "jams/hamiltonian/exchange.h"
#include "exchange.h"


using namespace std;

void ExchangeHamiltonian::insert_interaction(const int i, const int j, const Mat3 &value) {
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (std::abs(value[m][n]) * input_unit_conversion_ > energy_cutoff_ / kBohrMagneton) {
        interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n] * input_unit_conversion_);
      }
    }
  }
}

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {

  //---------------------------------------------------------------------
  // read settings
  //---------------------------------------------------------------------


  bool use_symops = true;
  settings.lookupValue("symops", use_symops);

  bool print_unfolded = false;
  settings.lookupValue("print_unfolded", print_unfolded);

  print_unfolded = print_unfolded || verbose_is_enabled() || debug_is_enabled();

  energy_cutoff_ = 1E-26;  // Joules
  settings.lookupValue("energy_cutoff", energy_cutoff_);
  cout << "    interaction energy cutoff " << energy_cutoff_ << "\n";

  radius_cutoff_ = 100.0;  // lattice parameters
  settings.lookupValue("radius_cutoff", radius_cutoff_);
  cout << "    interaction radius cutoff " << radius_cutoff_ << "\n";

  distance_tolerance_ = 1e-3; // fractional coordinate units
  settings.lookupValue("distance_tolerance", distance_tolerance_);
  cout << "    distance_tolerance " << distance_tolerance_ << "\n";

  safety_check_distance_tolerance(distance_tolerance_);

  if (debug_is_enabled()) {
    std::ofstream pos_file("debug_pos.dat");
    for (int n = 0; n < lattice->num_materials(); ++n) {
      for (int i = 0; i < globals::num_spins; ++i) {
        if (lattice->atom_material_id(i) == n) {
          pos_file << i << "\t" << lattice->atom_position(i) << " | "
                   << lattice->cartesian_to_fractional(lattice->atom_position(i)) << "\n";
        }
      }
      pos_file << "\n\n";
    }
    pos_file.close();
  }

  //---------------------------------------------------------------------
  // generate interaction list
  //---------------------------------------------------------------------
    std::string coordinate_format_name = "CARTESIAN";
    settings.lookupValue("coordinate_format", coordinate_format_name);
    CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

    cout << "    coordinate format: " << to_string(coord_format) << "\n";
    // exc_file is to maintain backwards compatibility
    if (settings.exists("exc_file")) {
      cout << "    interaction file name " << settings["exc_file"].c_str() << "\n";

      std::ifstream interaction_file(settings["exc_file"].c_str());
      if (interaction_file.fail()) {
        jams_die("failed to open interaction file");
      }

      neighbour_list_ = generate_neighbour_list(interaction_file, coord_format, use_symops, energy_cutoff_,
                                                radius_cutoff_);
    } else if (settings.exists("interactions")) {
      neighbour_list_ = generate_neighbour_list(settings["interactions"], coord_format, use_symops, energy_cutoff_, radius_cutoff_);
    }

    if (debug_is_enabled()) {
      std::ofstream debug_file("DEBUG_exchange_nbr_list.tsv");
      write_neighbour_list(debug_file, neighbour_list_);
      debug_file.close();
    }

    //---------------------------------------------------------------------
    // create sparse matrix
    //---------------------------------------------------------------------
   
    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);
    interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    cout << "    computed interactions: "<< neighbour_list_.num_interactions() << "\n";


    cout << "    interactions per motif position: \n";
    if (lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2) && !lattice->has_impurities()) {
        for (auto i = 0; i < lattice->num_motif_atoms(); ++i) {
            cout << "      " << i << ": " << neighbour_list_.num_interactions(i) <<"\n";
        }
    }

    for (int i = 0; i < neighbour_list_.size(); ++i) {
      for (auto const &j: neighbour_list_[i]) {
        insert_interaction(i, j.first, j.second);
      }
    }

    cout << "    converting interaction matrix format from MAP to CSR\n";
    interaction_matrix_.convertMAP2CSR();
    cout << "    exchange matrix memory (CSR): " << interaction_matrix_.calculateMemory() << " (MB)\n";
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_total_energy() {
  double total_energy = 0.0;

  #if HAS_OMP
  #pragma omp parallel for reduction(+:total_energy)
  #endif
    for (int i = 0; i < globals::num_spins; ++i) {
        total_energy += calculate_one_spin_energy(i);
    }

    return 0.5*total_energy;
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy(const int i) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double jij_sj[3] = {0.0, 0.0, 0.0};
    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = globals::s.data();

    for (auto m = 0; m < 3; ++m) {
      const auto begin = ptrb[3*i+m];
      const auto end = ptre[3*i+m];
      for (auto j = begin; j < end; ++j) {
        jij_sj[m] = jij_sj[m] + x[ indx[j] ]*val[j];
      }
    }
    return -(globals::s(i,0)*jij_sj[0] + globals::s(i,1)*jij_sj[1] + globals::s(i,2)*jij_sj[2]);
}

double ExchangeHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_field(i, local_field);

    e_initial = -(spin_initial[0]*local_field[0] + spin_initial[1]*local_field[1] + spin_initial[2]*local_field[2]);
    e_final = -(spin_final[0]*local_field[0] + spin_final[1]*local_field[1] + spin_final[2]*local_field[2]);

    return e_final - e_initial;
}

void ExchangeHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

void ExchangeHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    local_field[0] = 0.0, local_field[1] = 0.0; local_field[2] = 0.0;

    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = globals::s.data();

    for (auto m = 0; m < 3; ++m) {
      const auto begin = ptrb[3*i+m];
      const auto end = ptre[3*i+m];
      for (auto j = begin; j < end; ++j) {
        // k = indx[j];
        local_field[m] = local_field[m] + x[ indx[j] ]*val[j];
      }
    }
}

void ExchangeHamiltonian::calculate_fields() {
  assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);
  double one = 1.0;
  double zero = 0.0;
  const char transa[1] = {'N'};
  const char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
  const int num_rows = globals::num_spins3;
  const int num_cols = globals::num_spins3;

#ifdef HAS_MKL
    mkl_dcsrmv(transa, &num_rows, &num_cols, &one, matdescra, interaction_matrix_.valPtr(),
            interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(),
            &zero, field_.data());
#else
    jams::Xcsrmv(transa, num_rows, num_cols, 1.0, matdescra, interaction_matrix_.valPtr(),
      interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
}

const InteractionList<Mat3> &ExchangeHamiltonian::neighbour_list() const {
  return neighbour_list_;
}
