#include <set>

#include "core/globals.h"
#include "core/utils.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"



#include "hamiltonian/exchange.h"
// #include "hamiltonian/exchange_kernel.h"

namespace {
  struct vec4_compare {
    bool operator() (const jblib::Vec4<int>& lhs, const jblib::Vec4<int>& rhs) const {
      if (lhs.x < rhs.x) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y < rhs.y) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z) {
        return true;
      } else if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w < rhs.w) {
        return true;
      }
      return false;
    }
  };

}

bool ExchangeHamiltonian::insert_interaction(const int m, const int n, const jblib::Matrix<double, 3, 3> &value) {

  int counter = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (fabs(value[i][j]) > energy_cutoff_) {
        counter++;
        if(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(interaction_matrix_.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
            if(m >= n){
              interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
            }
          }else{
            if(m <= n){

              interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
            }
          }
        }else{
          interaction_matrix_.insertValue(3*m+i, 3*n+j, value[i][j]/mu_bohr_si);
        }
      }
    }
  }

  if (counter == 0) {
    return false;
  }

  return true;
}

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings) {

    // output in default format for now
    outformat_ = TEXT;

    if (!settings.exists("exc_file")) {
        jams_error("ExchangeHamiltonian: an exchange file must be specified");
    }

    std::string interaction_filename = settings["exc_file"];

    // read in typeA typeB rx ry rz Jij
    std::ifstream interaction_file(interaction_filename.c_str());

    if (interaction_file.fail()) {
      jams_error("failed to open interaction file %s", interaction_filename.c_str());
    }

    energy_cutoff_ = 1E-26;  // Joules
    if (settings.exists("energy_cutoff")) {
        energy_cutoff_ = settings["energy_cutoff"];
    }
    ::output.write("\ninteraction energy cutoff\n  %e\n", energy_cutoff_);

    ::output.write("\ninteraction vectors (%s)\n", interaction_filename.c_str());

    std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > int_interaction_list(lattice.num_motif_positions());

    read_interactions_with_symmetry(interaction_filename, int_interaction_list);

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);

    if (solver->is_cuda_solver()) {
#ifdef CUDA
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
      interaction_matrix_.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
#endif  //CUDA
    } else {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
    }

    ::output.write("\ncomputed interactions\n");

    bool is_all_inserts_successful = true;
    int counter = 0;
    // loop over the translation vectors for lattice size
    for (int i = 0; i < lattice.num_unit_cells(0); ++i) {
      for (int j = 0; j < lattice.num_unit_cells(1); ++j) {
        for (int k = 0; k < lattice.num_unit_cells(2); ++k) {
          // loop over atoms in the motif
          for (int m = 0, mend = lattice.num_motif_positions(); m < mend; ++m) {
            int local_site = lattice.site_index_by_unit_cell(i, j, k, m);

            std::vector<bool> is_already_interacting(globals::num_spins, false);
            is_already_interacting[local_site] = true;  // don't allow self interaction

            // loop over all possible interaction vectors
            for (int n = 0, nend = int_interaction_list[m].size(); n < nend; ++n) {

              jblib::Vec4<int> int_lookup_vec(
                i + int_interaction_list[m][n].first.x,
                j + int_interaction_list[m][n].first.y,
                k + int_interaction_list[m][n].first.z,
                (lattice.num_motif_positions() + m + int_interaction_list[m][n].first.w)%lattice.num_motif_positions());

              if (lattice.apply_boundary_conditions(int_lookup_vec) == false) {
                continue;
              }

              int neighbour_site = lattice.site_index_by_unit_cell(
                int_lookup_vec.x, int_lookup_vec.y, int_lookup_vec.z, int_lookup_vec.w);

              // failsafe check that we only interact with any given site once through the input exchange file.
              if (is_already_interacting[neighbour_site]) {
                jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_site, int_lookup_vec.x, int_lookup_vec.y, int_lookup_vec.z, int_lookup_vec.w);
              }
              is_already_interacting[neighbour_site] = true;

              if (insert_interaction(local_site, neighbour_site, int_interaction_list[m][n].second)) {
                // if(local_site >= neighbour_site) {
                //   std::cerr << local_site << "\t" << neighbour_site << "\t" << neighbour_site << "\t" << local_site << std::endl;
                // }
                counter++;
              } else {
                is_all_inserts_successful = false;
              }
              if (insert_interaction(neighbour_site, local_site, int_interaction_list[m][n].second)) {
                counter++;
              } else {
                is_all_inserts_successful = false;
              }
            }
          }
        }
      }
    }

    if (!is_all_inserts_successful) {
      jams_warning("Some interactions were ignored due to the energy cutoff (%e)", energy_cutoff_);
    }

    ::output.write("  total unit cell interactions: %d\n", counter);

    // resize member arrays
    energy_.resize(globals::num_spins);
    field_.resize(globals::num_spins, 3);


    // transfer arrays to cuda device if needed
    if (solver->is_cuda_solver()) {
#ifdef CUDA
        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_ = jblib::CudaArray<double, 1>(field_);

        ::output.write("  converting interaction matrix format from map to dia");
        interaction_matrix_.convertMAP2DIA();
        ::output.write("  estimated memory usage (DIA): %f MB\n", interaction_matrix_.calculateMemory());
        dev_interaction_matrix_.blocks = std::min<int>(DIA_BLOCK_SIZE, (globals::num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
        ::output.write("  allocating memory on device\n");

        // allocate rows
        CUDA_CALL(cudaMalloc((void**)&dev_interaction_matrix_.row, (interaction_matrix_.diags())*sizeof(int)));
        // allocate values
        CUDA_CALL(cudaMallocPitch((void**)&dev_interaction_matrix_.val, &dev_interaction_matrix_.pitch,
            (interaction_matrix_.rows())*sizeof(double), interaction_matrix_.diags()));
        // copy rows
        CUDA_CALL(cudaMemcpy(dev_interaction_matrix_.row, interaction_matrix_.dia_offPtr(),
            (size_t)((interaction_matrix_.diags())*(sizeof(int))), cudaMemcpyHostToDevice));
        // convert val array into double which may be float or double
        std::vector<double> float_values(interaction_matrix_.rows()*interaction_matrix_.diags(), 0.0);

        for (int i = 0; i < interaction_matrix_.rows()*interaction_matrix_.diags(); ++i) {
          float_values[i] = static_cast<double>(interaction_matrix_.val(i));
        }

        // copy values
        CUDA_CALL(cudaMemcpy2D(dev_interaction_matrix_.val, dev_interaction_matrix_.pitch, &float_values[0],
            interaction_matrix_.rows()*sizeof(double), interaction_matrix_.rows()*sizeof(double),
            interaction_matrix_.diags(), cudaMemcpyHostToDevice));

        dev_interaction_matrix_.pitch = dev_interaction_matrix_.pitch/sizeof(double);
#endif
    } else {
        ::output.write("  converting interaction matrix format from MAP to CSR\n");
        interaction_matrix_.convertMAP2CSR();
        ::output.write("  exchange matrix memory (CSR): %f MB\n", interaction_matrix_.calculateMemory());
    }

}

void ExchangeHamiltonian::read_interactions(const std::string &filename,
  std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list) {

  std::ifstream interaction_file(filename.c_str());

  if(interaction_file.fail()) {
    jams_error("failed to open interaction file %s", filename.c_str());
  }

  int_interaction_list.resize(lattice.num_motif_positions());


  int counter = 0;
  // read the motif into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {
    std::stringstream is(line);

    int typeA, typeB, type_difference;
    jblib::Vec3<double> interaction_vec_cart, interaction_vec_frac;

    is >> typeA >> typeB;
    typeA--; typeB--;  // zero base the types
    type_difference = (typeB - typeA);

    is >> interaction_vec_cart.x >> interaction_vec_cart.y >> interaction_vec_cart.z;
    // transform into lattice vector basis
    interaction_vec_frac = lattice.cartesian_to_fractional_position(interaction_vec_cart) + lattice.motif_position(typeA);

    // this 4-vector specifies the integer number of lattice vectors to the unit cell and the fourth
    // component is the atoms number within the motif
    jblib::Vec4<int> fast_integer_vector;
    for (int i = 0; i < 3; ++ i) {
      // rounding with nint accounts for lack of precision in definition of the real space vectors
      fast_integer_vector[i] = floor(interaction_vec_frac[i]+0.001);
    }
    fast_integer_vector[3] = type_difference;

    jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (file_columns(line) == 6) {
      // one Jij component given - diagonal
      is >> tensor[0][0];
      tensor[1][1] = tensor[0][0];
      tensor[2][2] = tensor[0][0];
    } else if (file_columns(line) == 14) {
      // nine Jij components given - full tensor
      is >> tensor[0][0] >> tensor[0][1] >> tensor[0][2];
      is >> tensor[1][0] >> tensor[1][1] >> tensor[1][2];
      is >> tensor[2][0] >> tensor[2][1] >> tensor[2][2];
    } else {
      jams_error("number of Jij values in exchange files must be 1 or 9, check your input on line %d", counter);
    }

    int_interaction_list[typeA].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(fast_integer_vector, tensor));

    if (verbose_output_is_set) {
      ::output.write("  line %-9d              %s\n", counter, line.c_str());
      ::output.write("  types               A : %d  B : %d  B - A : %d\n", typeA, typeB, type_difference);
      ::output.write("  interaction vector % 3.6f % 3.6f % 3.6f\n",
        interaction_vec_cart.x, interaction_vec_cart.y, interaction_vec_cart.z);
      ::output.write("  fractional vector  % 3.6f % 3.6f % 3.6f\n",
        interaction_vec_frac.x, interaction_vec_frac.y, interaction_vec_frac.z);
      ::output.write("  integer vector     % -9d % -9d % -9d % -9d\n\n",
        fast_integer_vector.x, fast_integer_vector.y, fast_integer_vector.z, fast_integer_vector.w);
    }
    counter++;
  }

  if(!verbose_output_is_set) {
    ::output.write("  ... [use verbose output for details] ... \n");
    ::output.write("  total: %d\n", counter);
  }

  interaction_file.close();



}

void ExchangeHamiltonian::read_interactions_with_symmetry(const std::string &filename,
  std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > &int_interaction_list) {

  verbose_output_is_set = true;

  std::ifstream interaction_file(filename.c_str());

  if(interaction_file.fail()) {
    jams_error("failed to open interaction file %s", filename.c_str());
  }

  int_interaction_list.resize(lattice.num_motif_positions());

  int counter = 0;
  // read the motif into an array from the positions file
  for (std::string line; getline(interaction_file, line); ) {
    std::stringstream is(line);

    std::string type_name_A, type_name_B;

    // read type names
    is >> type_name_A >> type_name_B;

    jblib::Vec3<double> interaction_vector;
    is >> interaction_vector.x >> interaction_vector.y >> interaction_vector.z;

    jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (file_columns(line) == 6) {
      // one Jij component given - diagonal
      is >> tensor[0][0];
      tensor[1][1] = tensor[0][0];
      tensor[2][2] = tensor[0][0];
    } else if (file_columns(line) == 14) {
      // nine Jij components given - full tensor
      is >> tensor[0][0] >> tensor[0][1] >> tensor[0][2];
      is >> tensor[1][0] >> tensor[1][1] >> tensor[1][2];
      is >> tensor[2][0] >> tensor[2][1] >> tensor[2][2];
    } else {
      jams_error("number of Jij values in exchange files must be 1 or 9, check your input on line %d", counter);
    }

    // transform into lattice vector basis
    // jblib::Vec3<double> lattice_vector = (inverse_lattice_vectors_*interaction_vector);

    jblib::Vec3<double> r(interaction_vector.x, interaction_vector.y, interaction_vector.z);
    jblib::Vec3<double> r_sym;

    jblib::Array<double, 2> sym_interaction_vecs(lattice.num_sym_ops(), 3);

    ::output.verbose("\ninteraction vectors (%s)\n", filename.c_str());

    for (int i = 0; i < lattice.num_sym_ops(); i++) {
      jblib::Vec3<double> rij(r[0], r[1], r[2]);
      rij = lattice.cartesian_to_fractional_position(r);
      r_sym = lattice.sym_rotation(i)*rij;
      for (int j = 0; j < 3; ++j) {
        sym_interaction_vecs(i,j) = r_sym[j]; // + spglib_dataset_->translations[i][j];
      }
      // ::output.verbose("%f %f %f\n", r_sym[0], r_sym[1], r_sym[2]);
    }

    if (verbose_output_is_set) {
      ::output.verbose("unit cell realspace\n");
      for (int i = 0; i < lattice.num_motif_positions(); ++i) {
        jblib::Vec3<double> rij = lattice.motif_position(i);
        ::output.verbose("%8d % 6.6f % 6.6f % 6.6f\n", i, rij[0], rij[1], rij[2]);
      }
    }

    ::output.verbose("unit cell interactions\n");
    for (int i = 0; i < lattice.num_motif_positions(); ++i) {
      std::set<jblib::Vec4<int>, vec4_compare> interaction_set;
      // only process for interactions belonging to this material
      if (lattice.motif_material(i) == type_name_A) {
        ::output.verbose("motif position %d\n", i);
        ::output.verbose("        i         j ::             rij             |             uij             |             r0              |          motif offset          |          real rij          |         real offset\n");

        // motif position in basis vector space
        jblib::Vec3<double> pij = lattice.fractional_to_cartesian_position(lattice.motif_position(i));

        for (int j = 0; j < sym_interaction_vecs.size(0); ++ j) {

          // position of neighbour in real space
          jblib::Vec3<double> rij(sym_interaction_vecs(j,0), sym_interaction_vecs(j,1), sym_interaction_vecs(j,2));

          rij = lattice.cartesian_to_fractional_position(rij) + pij;
          // for (int k = 0; k < 3; ++k) {
          //   rij[k] = sym_interaction_vecs(j,k) + pij[k];
          // }

          ::output.verbose("% 6.6f % 6.6f % 6.6f | ", rij[0], rij[1], rij[2]);

          jblib::Vec3<double> rij_inv = lattice.cartesian_to_fractional_position(rij);

          ::output.verbose("% 6.6f % 6.6f % 6.6f | ", rij_inv[0], rij_inv[1], rij_inv[2]);

          ::output.verbose("% 6.6f % 6.6f % 6.6f\n", floor(rij_inv[0]), floor(rij_inv[1]), floor(rij_inv[2]));


          // jblib::Vec3<double> real_rij = lattice_vectors_*(rij + pij);


          // integer unit cell index containing the neighbour
          jblib::Vec3<double> uij = rij + pij;
          for (int k = 0; k < 3; ++k) {
            uij[k] = floor(uij[k]);
          }

          // origin of unit cell containing the neighbour in basis vector space
          jblib::Vec3<double> r0 = uij;

          // jblib::Vec3<double> motif_offset = rij - uij;
          jblib::Vec3<double> motif_pos_offset = (rij + pij) - uij;

          jblib::Vec3<double> real_offset = lattice.fractional_to_cartesian_position(motif_pos_offset);


          // // now calculate the motif offset
          // double motif_offset[3];
          // for (int k = 0; k < 3; ++k) {
          //   motif_offset[k] = (sym_interaction_vecs(j,k) + motif_[i].second[k]) - fast_integer_vector[k];
          // }
          // find which motif position this offset corresponds to
          // it is possible that it does not correspond to a position in which case the
          // "motif_partner" is -1
          int motif_partner = -1;
          for (int k = 0; k < lattice.num_motif_positions(); ++k) {
            jblib::Vec3<double> pos = lattice.motif_position(k);
            if ( fabs(pos.x - motif_pos_offset.x) < 1e-4
              && fabs(pos.y - motif_pos_offset.y) < 1e-4
              && fabs(pos.z - motif_pos_offset.z) < 1e-4 ) {
              motif_partner = k;
              break;
            }
          }

          // ::output.verbose("% 8d % 8d :: % 6.6f % 6.6f % 6.6f | % 8d % 8d % 8d | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f | % 6.6f % 6.6f % 6.6f\n",
            // i, motif_partner, rij[0], rij[1], rij[2], int(uij[0]), int(uij[1]), int(uij[2]),
            // r0[0], r0[1], r0[2], motif_offset[0], motif_offset[1], motif_offset[2], real_rij[0], real_rij[1], real_rij[2], real_offset[0], real_offset[1], real_offset[2]);

          if (motif_partner != -1) {
            // fast_integer_vector[3] = motif_partner - i;
            // check the motif partner is of the type that was specified in the exchange input file
            if (lattice.motif_material(motif_partner) != type_name_B) {
              ::output.write("wrong type ");
            }
            jblib::Vec4<int> fast_integer_vector(uij[0], uij[1], uij[2], motif_partner - i);
            // ::output.write("%d %d %d %d %d % 3.6f % 3.6f % 3.6f\n", i, motif_partner, fast_integer_vector[0], fast_integer_vector[1], fast_integer_vector[2], motif_offset[0], motif_offset[1], motif_offset[2]);
            interaction_set.insert(fast_integer_vector);
          }
        }
        for (std::set<jblib::Vec4<int> >::iterator it = interaction_set.begin(); it != interaction_set.end(); ++it) {
          ::output.verbose("% 8d % 8d :: % 8d % 8d % 8d\n", i, (*it)[3] + i, (*it)[0], (*it)[1], (*it)[2]);
          int_interaction_list[i].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >((*it), tensor));
          counter++;
        }
      }
    }
  }
  ::output.write("  total unit cell interactions: %d\n", counter);
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_total_energy() {
    return 0.0;
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double jij_sj[3] = {0.0, 0.0, 0.0};
    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();
    int           k;

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) {
        k = indx[j];
        jij_sj[m] = jij_sj[m] + x[k]*val[j];
      }
    }
    return -(s(i,0)*jij_sj[0] + s(i,1)*jij_sj[1] + s(i,2)*jij_sj[2]);
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_fields(i, local_field);
    e_initial = -(spin_initial[0]*local_field[0] + spin_initial[1]*local_field[1] + spin_initial[2]*local_field[2]);
    e_final = -(spin_final[0]*local_field[0] + spin_final[1]*local_field[1] + spin_final[2]*local_field[2]);

    return e_final - e_initial;
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_one_spin_fields(const int i, double local_field[3]) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    local_field[0] = 0.0, local_field[1] = 0.0; local_field[2] = 0.0;

    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();
    int k, j, m, begin, end;

    for (m = 0; m < 3; ++m) {
      begin = ptrb[3*i+m]; end = ptre[3*i+m];
      for (j = begin; j < end; ++j) {
        k = indx[j];
        local_field[m] = local_field[m] + x[k]*val[j];
      }
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_fields() {

    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
        spmv_dia_kernel<<< dev_interaction_matrix_.blocks, DIA_BLOCK_SIZE >>>
            (globals::num_spins3, globals::num_spins3, interaction_matrix_.diags(), dev_interaction_matrix_.pitch, 1.0, 0.0,
            dev_interaction_matrix_.row, dev_interaction_matrix_.val, solver->dev_ptr_spin(), dev_field_.data());
#endif  // CUDA
    } else {
      if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL) {
        // general matrix (i.e. Monte Carlo Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
        double one = 1.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &one, globals::h.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 1.0, globals::h.data());
#endif
      } else {
        // symmetric matrix (i.e. Heun Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef MKL
        double one = 1.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &one, globals::h.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 1.0, globals::h.data());
#endif
      }
    }
}
// --------------------------------------------------------------------------

void ExchangeHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("Exchange energy output: HDF5 not yet implemented");
        default:
            jams_error("Exchange energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("Exchange energy output: HDF5 not yet implemented");
        default:
            jams_error("Exchange energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::output_energies_text() {
    using namespace globals;

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_energy_.copy_to_host_array(energy_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_eng_uniaxial_"+zero_pad_number(outcount)+".dat");

    std::ofstream outfile(filename.c_str());

    outfile << "# type | rx (nm) | ry (nm) | rz (nm) | d2z | d4z | d6z" << std::endl;

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << lattice.lattice_material_num_[i];

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile <<  lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
        }

        // energy

    }
    outfile.close();
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::output_fields_text() {

#ifdef CUDA
    if (globals::is_cuda_solver_used) {
        dev_field_.copy_to_host_array(field_);
    }
#endif  // CUDA

    int outcount = 0;

    const std::string filename(seedname+"_field_uniaxial_"+zero_pad_number(outcount)+".dat");

    // using direct file access for performance
    std::ofstream outfile(filename.c_str());
    outfile.setf(std::ios::right);

    outfile << "#";
    outfile << std::setw(16) << "type";
    outfile << std::setw(16) << "rx (nm)";
    outfile << std::setw(16) << "ry (nm)";
    outfile << std::setw(16) << "rz (nm)";
    outfile << std::setw(16) << "hx (nm)";
    outfile << std::setw(16) << "hy (nm)";
    outfile << std::setw(16) << "hz (nm)";
    outfile << "\n";

    for (int i = 0; i < globals::num_spins; ++i) {
        // spin type
        outfile << std::setw(16) << lattice.lattice_material_num_[i];

        // real position
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::fixed << lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
        }

        // fields
        for (int j = 0; j < 3; ++j) {
            outfile << std::setw(16) << std::scientific << field_(i,j);
        }
        outfile << "\n";
    }
    outfile.close();
}