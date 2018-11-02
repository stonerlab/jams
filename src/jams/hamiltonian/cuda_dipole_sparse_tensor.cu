#include <cmath>

#include <libconfig.h++>

#include "jams/interface/blas.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/hamiltonian/cuda_dipole_sparse_tensor.h"

using namespace std;

CudaDipoleHamiltonianSparseTensor::~CudaDipoleHamiltonianSparseTensor() {
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}

CudaDipoleHamiltonianSparseTensor::CudaDipoleHamiltonianSparseTensor(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size),
    use_double_precision(false)   // default to float precision
 {
    using std::pow;
    double r_abs;
    Vec3 r_hat;

    cout << "  strategy: cuda_sparse_tensor\n";

    r_cutoff_ = settings["r_cutoff"];

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);


    if (solver->is_cuda_solver()) {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
      interaction_matrix_.setMatrixMode(SPARSE_FILL_MODE_LOWER);
    } else {
      interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);
    }
    
    const double prefactor = kVacuumPermeadbility*kBohrMagneton/(4*kPi*pow(::lattice->parameter(),3));

    Mat3 Id = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    cout << "    precalculating number of interactions\n";
    unsigned int interaction_count = 0;
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {

            if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
              if (j > i) {
                continue;
              }
            }

            if (j == i) continue;

            auto r_ij = lattice->displacement(lattice->atom_position(i), lattice->atom_position(j));

            r_abs = abs(r_ij);

            // i can interact with i in another image of the simulation cell (just not the 0, 0, 0 image)
            // so detect based on r_abs rather than i == j
            if (r_abs > r_cutoff_ || unlikely(r_abs < 1e-5)) continue;

            interaction_count++;
      }
    }

    cout << "    interaction count " << interaction_count << "\n";

    cout << "    reserving memory for sparse matrix " << 9*interaction_count*(sizeof(uint64_t)+sizeof(float))/(1024.0*1024.0) << "(MB)\n";
    interaction_matrix_.reserveMemory(9*interaction_count); // 9 elements in tensor

    cout << "    inserting tensor elements\n";
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {

            if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
              if (j > i) {
                continue;
              }
            }

            if (j == i) continue;

          auto r_ij = lattice->displacement(lattice->atom_position(i), lattice->atom_position(j));

            r_abs = abs(r_ij);

            // i can interact with i in another image of the simulation cell (just not the 0, 0, 0 image)
            // so detect based on r_abs rather than i == j
            if (r_abs > r_cutoff_ || unlikely(r_abs < 1e-5)) continue;

            r_hat = r_ij / r_abs;

            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    double value = (3*r_hat[m]*r_hat[n] - Id[m][n])*prefactor*globals::mus(i)*globals::mus(j)/(r_abs * r_abs * r_abs);
                    interaction_matrix_.insertValue(3 * i + m, 3 * j + n, float(value));
                }
            }
      }
    }


    cout << "    dipole matrix memory (MAP) " << interaction_matrix_.calculateMemory() << "(MB)\n";
    cout << "    converting interaction matrix format from MAP to CSR\n";
    interaction_matrix_.convertMAP2CSR();
    cout << "    dipole matrix memory (CSR) " << interaction_matrix_.calculateMemory() << " (MB)\n";

    // set up things on the device
    if (solver->is_cuda_solver()) { 
        cudaStreamCreate(&dev_stream_);

        cusparseStatus_t cusparse_return_status;


        cout << "    initialising CUSPARSE\n";
        cusparse_return_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_return_status != CUSPARSE_STATUS_SUCCESS) {
          die("CUSPARSE Library initialization failed");
        }
        cusparseSetStream(cusparse_handle_, dev_stream_);


        cusparse_return_status = cusparseCreateMatDescr(&cusparse_descra_);
        if (cusparse_return_status != CUSPARSE_STATUS_SUCCESS) {
          die("CUSPARSE Matrix descriptor initialization failed");
        }

        if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL) {
          cusparseSetMatType(cusparse_descra_,CUSPARSE_MATRIX_TYPE_GENERAL);
        } else if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
          cusparseSetMatType(cusparse_descra_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
        } else {
          die("unknown sparse matrix type in dipole_cuda_sparse_tensor");
        }
        cusparseSetMatIndexBase(cusparse_descra_, CUSPARSE_INDEX_BASE_ZERO);

        // row
        cout << "    allocating csr row on device\n";
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.row, (interaction_matrix_.rows()+1)*sizeof(int)));
        
        cout << "    memcpy csr row to device\n";
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.row, interaction_matrix_.rowPtr(),
              (interaction_matrix_.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

        // col
        cout << "    allocating csr col on device\n";
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.col, (interaction_matrix_.nonZero())*sizeof(int)));

        cout << "    memcpy csr col to device\n";
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.col, interaction_matrix_.colPtr(),
              (interaction_matrix_.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

        // val
        cout << "    allocating csr val on device\n";
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.val, (interaction_matrix_.nonZero())*sizeof(float)));

        cout << "    memcpy csr val to device\n";
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.val, interaction_matrix_.valPtr(),
              (interaction_matrix_.nonZero())*sizeof(float), cudaMemcpyHostToDevice));

        dev_float_spins_.resize(globals::num_spins3);
        dev_float_fields_.resize(globals::num_spins3);

    }
}

// --------------------------------------------------------------------------

double CudaDipoleHamiltonianSparseTensor::calculate_total_energy() {
   double e_total = 0.0;
   for (int i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_one_spin_energy(i);
   }
    return 0.5*e_total;
}

// --------------------------------------------------------------------------


double CudaDipoleHamiltonianSparseTensor::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double CudaDipoleHamiltonianSparseTensor::calculate_one_spin_energy(const int i) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double wij_sj[3] = {0.0, 0.0, 0.0};
    const float *val = interaction_matrix_.valPtr();
    const int    *indx = interaction_matrix_.colPtr();
    const int    *ptrb = interaction_matrix_.ptrB();
    const int    *ptre = interaction_matrix_.ptrE();
    const double *x   = s.data();

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) {
        wij_sj[m] = wij_sj[m] + x[ indx[j] ]*val[j];
      }
    }
    return -globals::mus(i)*(s(i,0)*wij_sj[0] + s(i,1)*wij_sj[1] + s(i,2)*wij_sj[2]);
}

// --------------------------------------------------------------------------

double CudaDipoleHamiltonianSparseTensor::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    double local_field[3], e_initial, e_final;

    calculate_one_spin_field(i, local_field);

    e_initial = -(spin_initial[0]*local_field[0] + spin_initial[1]*local_field[1] + spin_initial[2]*local_field[2]);
    e_final = -(spin_final[0]*local_field[0] + spin_final[1]*local_field[1] + spin_final[2]*local_field[2]);

    return 0.5 * (e_final - e_initial);
}
// --------------------------------------------------------------------------

void CudaDipoleHamiltonianSparseTensor::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void CudaDipoleHamiltonianSparseTensor::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    assert(interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);

    local_field[0] = 0.0, local_field[1] = 0.0; local_field[2] = 0.0;

    const float *val = interaction_matrix_.valPtr();
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

      local_field[m] = local_field[m]*globals::mus(i);
    }


}


// --------------------------------------------------------------------------

void CudaDipoleHamiltonianSparseTensor::calculate_fields(jblib::Array<double, 2>& fields) {
  if (interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL) {
    // general matrix (i.e. Monte Carlo Solvers)
      char transa[1] = {'N'};
      char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};

      jams_scsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
        interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, fields.data());
    } else {
      // symmetric matrix (i.e. Heun Solvers)
      char transa[1] = {'N'};
      char matdescra[6] = {'S', 'L', 'N', 'C', 'N', 'N'};
      jams_scsrmv(transa, globals::num_spins3, globals::num_spins3, 1.0, matdescra, interaction_matrix_.valPtr(),
        interaction_matrix_.colPtr(), interaction_matrix_.ptrB(), interaction_matrix_.ptrE(), globals::s.data(), 0.0, fields.data());
    }
}

void CudaDipoleHamiltonianSparseTensor::calculate_fields(jblib::CudaArray<double, 1>& fields) {

    // cast spin array to floats
    cuda_array_double_to_float(globals::num_spins3, solver->dev_ptr_spin(), dev_float_spins_.data(), dev_stream_);

    const float one = 1.0;
    const float zero = 0.0;
    cusparseStatus_t stat =
    cusparseScsrmv(cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      globals::num_spins3,
      globals::num_spins3,
      interaction_matrix_.nonZero(),
      &one,
      cusparse_descra_,
      dev_csr_interaction_matrix_.val,
      dev_csr_interaction_matrix_.row,
      dev_csr_interaction_matrix_.col,
      dev_float_spins_.data(),
      &zero,
      dev_float_fields_.data());
    assert(stat == CUSPARSE_STATUS_SUCCESS);

    cuda_array_float_to_double(globals::num_spins3, dev_float_fields_.data(), fields.data(), dev_stream_);
}

// --------------------------------------------------------------------------
