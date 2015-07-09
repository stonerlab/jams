#include "core/globals.h"
#include "core/utils.h"
#include "core/consts.h"
#include "core/cuda_defs.h"
#include "core/cuda_sparsematrix.h"



#include "hamiltonian/exchange.h"
// #include "hamiltonian/exchange_kernel.h"

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

    std::vector< std::vector< std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> > > > fast_integer_interaction_list_(lattice.num_motif_positions());

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

      fast_integer_interaction_list_[typeA].push_back(std::pair<jblib::Vec4<int>, jblib::Matrix<double, 3, 3> >(fast_integer_vector, tensor));

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

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);
    interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    interaction_matrix_.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);

    ::output.write("\ncomputed interactions\n");

    bool is_all_inserts_successful = true;
    counter = 0;
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
            for (int n = 0, nend = fast_integer_interaction_list_[m].size(); n < nend; ++n) {

              jblib::Vec4<int> fast_integer_lookup_vector(
                i + fast_integer_interaction_list_[m][n].first.x,
                j + fast_integer_interaction_list_[m][n].first.y,
                k + fast_integer_interaction_list_[m][n].first.z,
                (lattice.num_motif_positions() + m + fast_integer_interaction_list_[m][n].first.w)%lattice.num_motif_positions());

              bool interaction_is_outside_lattice = false;
              // if we are trying to interact with a site outside of the boundary
              for (int l = 0; l < 3; ++l) {
                if (lattice.is_periodic(l)) {
                  fast_integer_lookup_vector[l] = (fast_integer_lookup_vector[l] + lattice.num_unit_cells(l))%lattice.num_unit_cells(l);
                } else {
                  if (fast_integer_lookup_vector[l] < 0 || fast_integer_lookup_vector[l] >= lattice.num_unit_cells(l)) {
                    interaction_is_outside_lattice = true;
                  }
                }
              }
              if (interaction_is_outside_lattice) {
                continue;
              }

              int neighbour_site = lattice.site_index_by_unit_cell(fast_integer_lookup_vector.x, fast_integer_lookup_vector.y, fast_integer_lookup_vector.z, fast_integer_lookup_vector.w);

              // failsafe check that we only interact with any given site once through the input exchange file.
              if (is_already_interacting[neighbour_site]) {
                jams_error("Multiple interactions between spins %d and %d.\nInteger vectors %d  %d  %d  %d\nCheck the exchange file.", local_site, neighbour_site, fast_integer_lookup_vector.x, fast_integer_lookup_vector.y, fast_integer_lookup_vector.z, fast_integer_lookup_vector.w);
              }
              is_already_interacting[neighbour_site] = true;

              if (insert_interaction(local_site, neighbour_site, fast_integer_interaction_list_[m][n].second)) {
                // if(local_site >= neighbour_site) {
                //   std::cerr << local_site << "\t" << neighbour_site << "\t" << neighbour_site << "\t" << local_site << std::endl;
                // }
                counter++;
              } else {
                is_all_inserts_successful = false;
              }
              if (insert_interaction(neighbour_site, local_site, fast_integer_interaction_list_[m][n].second)) {
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

    ::output.write("  total: %d\n", counter);


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

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_total_energy() {
    return 0.0;
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    return 0.0;
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_one_spin_fields(const int i, double h[3]) {
    using namespace globals;
    h[0] = 0.0; h[1] = 0.0;
    h[2] = 0.0;
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
          if (interaction_matrix_.nonZero() > 0) {
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
