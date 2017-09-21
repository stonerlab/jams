#include <set>
#include <tuple>
#include <stdexcept>

#include "jams/core/exception.h"
#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/cuda_defs.h"
#include "jams/core/cuda_sparsematrix.h"
#include "jams/core/interactions.h"
#include "jams/core/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "jblib/math/summations.h"

#include "jams/hamiltonian/exchange.h"

namespace {
    InteractionFileFormat exchange_file_format_from_string(std::string s) {
      if (capitalize(s) == "JAMS") {
        return InteractionFileFormat::JAMS;
      }

      if (capitalize(s) == "KKR") {
        return InteractionFileFormat::KKR;
      }

      throw std::runtime_error("Unknown exchange file format");
    }

    inline CoordinateFormat coordinate_format_from_string(std::string s) {
      if (capitalize(s) == "CART" || capitalize(s) == "CARTESIAN") {
        return CoordinateFormat::Cartesian;
      }

      if (capitalize(s) == "FRAC" || capitalize(s) == "FRACTIONAL") {
        return CoordinateFormat::Fractional;
      }

      throw std::runtime_error("Unknown coordinate format");
    }

}

ExchangeHamiltonian::~ExchangeHamiltonian() {
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}

//---------------------------------------------------------------------

void ExchangeHamiltonian::insert_interaction(const int i, const int j, const Mat3 &value) {
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      if (std::abs(value[m][n]) > energy_cutoff_) {
        if(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          if(interaction_matrix_.getMatrixMode() == SPARSE_FILL_MODE_LOWER) {
            if (i >= j) {
              interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
            }
          } else {
            if (i <= j) {
              interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
            }
          }
        } else {
          interaction_matrix_.insertValue(3*i+m, 3*j+n, value[m][n]);
        }
      }
    }
  }
}

//---------------------------------------------------------------------

ExchangeHamiltonian::ExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {
  ::output->write("initialising Hamiltonian: %s\n", this->name().c_str());

  //---------------------------------------------------------------------
  // read settings
  //---------------------------------------------------------------------

    is_debug_enabled_ = false;
    settings.lookupValue("debug", is_debug_enabled_);

    std::string interaction_filename = settings["exc_file"].c_str();
    std::ifstream interaction_file(interaction_filename.c_str());
    if (interaction_file.fail()) {
      jams_error("failed to open interaction file %s", interaction_filename.c_str());
    }
    ::output->write("\ninteraction filename (%s)\n", interaction_filename.c_str());

    std::string exchange_file_format_name = "JAMS";
    settings.lookupValue("format", exchange_file_format_name);
    exchange_file_format_ = exchange_file_format_from_string(exchange_file_format_name);

    std::string coordinate_format_name = "CARTESIAN";
    settings.lookupValue("coordinate_format", coordinate_format_name);
    CoordinateFormat coord_format = coordinate_format_from_string(coordinate_format_name);

    bool use_symops = true;
    settings.lookupValue("symops", use_symops);

    bool print_unfolded = false;
    settings.lookupValue("print_unfolded", print_unfolded);

    energy_cutoff_ = 1E-26;  // Joules
    settings.lookupValue("energy_cutoff", energy_cutoff_);
    ::output->write("\ninteraction energy cutoff\n  %e\n", energy_cutoff_);

    radius_cutoff_ = 100.0;  // lattice parameters
    settings.lookupValue("radius_cutoff", radius_cutoff_);
    ::output->write("\ninteraction radius cutoff\n  %e\n", radius_cutoff_);

    distance_tolerance_ = 1e-3; // fractional coordinate units
    settings.lookupValue("distance_tolerance", distance_tolerance_);
    ::output->write("\ndistance_tolerance\n  %e\n", distance_tolerance_);
    
    safety_check_distance_tolerance(distance_tolerance_);

    if (is_debug_enabled_) {
      std::ofstream pos_file("debug_pos.dat");
      for (int n = 0; n < lattice->num_materials(); ++n) {
        for (int i = 0; i < globals::num_spins; ++i) {
          if (lattice->atom_material(i) == n) {
            pos_file << i << "\t" <<  lattice->atom_position(i) << " | " << lattice->cartesian_to_fractional(lattice->atom_position(i)) << "\n";
          }
        }
        pos_file << "\n\n";
      }
      pos_file.close();
    }

    //---------------------------------------------------------------------
    // generate interaction list
    //---------------------------------------------------------------------
  generate_neighbour_list_from_file(interaction_file, exchange_file_format_, coord_format, energy_cutoff_, radius_cutoff_, use_symops,
                                    print_unfolded || is_debug_enabled_, neighbour_list_);

    if (is_debug_enabled_) {
      std::ofstream debug_file("DEBUG_exchange_nbr_list.tsv");
      write_neighbour_list(debug_file, neighbour_list_);
      debug_file.close();
    }

    //---------------------------------------------------------------------
    // create sparse matrix
    //---------------------------------------------------------------------
   
    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);

    if (solver->is_cuda_solver()) {
#ifdef CUDA
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
      // interaction_matrix_.setMatrixMode(SPARSE_FILL_MODE_LOWER);
#endif  //CUDA
    } else {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
    }

    ::output->write("\ncomputed interactions\n");

    for (int i = 0; i < neighbour_list_.size(); ++i) {
      for (auto const &j: neighbour_list_[i]) {
        insert_interaction(i, j.first, j.second);
      }
    }

    ::output->write("  converting interaction matrix format from MAP to CSR\n");
    interaction_matrix_.convertMAP2CSR();
    ::output->write("  exchange matrix memory (CSR): %f MB\n", interaction_matrix_.calculateMemory());

    // transfer arrays to cuda device if needed
    if (solver->is_cuda_solver()) {
#ifdef CUDA

        cudaStreamCreate(&dev_stream_);

        dev_energy_ = jblib::CudaArray<double, 1>(energy_);
        dev_field_  = jblib::CudaArray<double, 1>(field_);

        if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR) {
          ::output->write("  * Initialising CUSPARSE...\n");
          cusparseStatus_t status;
          status = cusparseCreate(&cusparse_handle_);
          if (status != CUSPARSE_STATUS_SUCCESS) {
            jams_error("CUSPARSE Library initialization failed");
          }
          cusparseSetStream(cusparse_handle_, dev_stream_);

          sparsematrix_copy_host_csr_to_cuda_csr(interaction_matrix_, dev_csr_interaction_matrix_);

        } else if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_DIA) {
          // ::output->write("  converting interaction matrix format from map to dia");
          // interaction_matrix_.convertMAP2DIA();
          ::output->write("  estimated memory usage (DIA): %f MB\n", interaction_matrix_.calculateMemory());
          dev_dia_interaction_matrix_.blocks = std::min<int>(DIA_BLOCK_SIZE, (globals::num_spins3+DIA_BLOCK_SIZE-1)/DIA_BLOCK_SIZE);
          ::output->write("  allocating memory on device\n");

          // allocate rows
          cuda_api_error_check(
            cudaMalloc((void**)&dev_dia_interaction_matrix_.row, (interaction_matrix_.diags())*sizeof(int)));
          // allocate values
          cuda_api_error_check(
            cudaMallocPitch((void**)&dev_dia_interaction_matrix_.val, &dev_dia_interaction_matrix_.pitch,
              (interaction_matrix_.rows())*sizeof(double), interaction_matrix_.diags()));
          // copy rows
          cuda_api_error_check(
            cudaMemcpy(dev_dia_interaction_matrix_.row, interaction_matrix_.dia_offPtr(),
              (size_t)((interaction_matrix_.diags())*(sizeof(int))), cudaMemcpyHostToDevice));
          // convert val array into double which may be float or double
          std::vector<double> float_values(interaction_matrix_.rows()*interaction_matrix_.diags(), 0.0);

          for (int i = 0; i < interaction_matrix_.rows()*interaction_matrix_.diags(); ++i) {
            float_values[i] = static_cast<double>(interaction_matrix_.val(i));
          }

          // copy values
          cuda_api_error_check(
            cudaMemcpy2D(dev_dia_interaction_matrix_.val, dev_dia_interaction_matrix_.pitch, &float_values[0],
              interaction_matrix_.rows()*sizeof(double), interaction_matrix_.rows()*sizeof(double),
              interaction_matrix_.diags(), cudaMemcpyHostToDevice));

          dev_dia_interaction_matrix_.pitch = dev_dia_interaction_matrix_.pitch/sizeof(double);
        }
#endif
  }

}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_total_energy() {
  double total_energy;

#ifdef CUDA
  if (solver->is_cuda_solver()) {
    calculate_fields();
    dev_field_.copy_to_host_array(field_);
    for (int i = 0; i < globals::num_spins; ++i) {
        total_energy += -(  globals::s(i,0)*field_(i,0) 
                     + globals::s(i,1)*field_(i,1)
                     + globals::s(i,2)*field_(i,2) );
    }
  } else {
#endif // CUDA

    for (int i = 0; i < globals::num_spins; ++i) {
        total_energy += calculate_one_spin_energy(i);
    }

#ifdef CUDA
    }
#endif // CUDA

    return 0.5*total_energy;
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

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) {
        jij_sj[m] = jij_sj[m] + x[ indx[j] ]*val[j];
      }
    }
    return -(s(i,0)*jij_sj[0] + s(i,1)*jij_sj[1] + s(i,2)*jij_sj[2]);
}

// --------------------------------------------------------------------------

double ExchangeHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_field(i, local_field);

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

void ExchangeHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    local_field[0] = 0.0, local_field[1] = 0.0; local_field[2] = 0.0;

    const double *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();
    int j, m, begin, end;

    for (m = 0; m < 3; ++m) {
      begin = ptrb[3*i+m]; end = ptre[3*i+m];
      for (j = begin; j < end; ++j) {
        // k = indx[j];
        local_field[m] = local_field[m] + x[ indx[j] ]*val[j];
      }
    }
}

// --------------------------------------------------------------------------

void ExchangeHamiltonian::calculate_fields() {
    // dev_s needs to be found from the solver

    if (solver->is_cuda_solver()) {
#ifdef CUDA
      if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_CSR) {
        const double one = 1.0;
        const double zero = 0.0;
        cusparseStatus_t stat =
        cusparseDcsrmv(cusparse_handle_,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          globals::num_spins3,
          globals::num_spins3,
          interaction_matrix_.nonZero(),
          &one,
          dev_csr_interaction_matrix_.descr,
          dev_csr_interaction_matrix_.val,
          dev_csr_interaction_matrix_.row,
          dev_csr_interaction_matrix_.col,
          solver->dev_ptr_spin(),
          &zero,
          dev_field_.data());
        assert(stat == CUSPARSE_STATUS_SUCCESS);
      } else if (interaction_matrix_.getMatrixFormat() == SPARSE_MATRIX_FORMAT_DIA) {
        spmv_dia_kernel<<< dev_dia_interaction_matrix_.blocks, DIA_BLOCK_SIZE >>>
            (globals::num_spins3, globals::num_spins3, interaction_matrix_.diags(), dev_dia_interaction_matrix_.pitch, 1.0, 0.0,
            dev_dia_interaction_matrix_.row, dev_dia_interaction_matrix_.val, solver->dev_ptr_spin(), dev_field_.data());
      }
#endif  // CUDA
    } else {
      if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL) {
        // general matrix (i.e. Monte Carlo Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
#ifdef USE_MKL
        double one = 1.0;
        double zero = 0.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &zero, field_.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
      } else {
        // symmetric matrix (i.e. Heun Solvers)
        char transa[1] = {'N'};
        char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
#ifdef USE_MKL
        double one = 1.0;
        double zero = 0.0;
        mkl_dcsrmv(transa, &globals::num_spins3, &globals::num_spins3, &one, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), &zero, field_.data());
#else
        jams_dcsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
          interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, field_.data());
#endif
      }
    }
}

// --------------------------------------------------------------------------

sparse_matrix_format_t ExchangeHamiltonian::sparse_matrix_format() {
  return interaction_matrix_format_;
}

//---------------------------------------------------------------------

void ExchangeHamiltonian::set_sparse_matrix_format(std::string &format_name) {
  if (capitalize(format_name) == "CSR") {
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_CSR;
  } else if (capitalize(format_name) == "DIA") {
    if (solver->is_cuda_solver() != true) {
      jams_error("ExchangeHamiltonian::set_sparse_matrix_format: DIA format is only supported for CUDA");
    }
    interaction_matrix_format_ = SPARSE_MATRIX_FORMAT_DIA;
  } else {
    jams_error("ExchangeHamiltonian::set_sparse_matrix_format: Unknown format requested %s", format_name.c_str());
  }
}

