#include <cmath>

#include <cblas.h>
#include "core/cuda_defs.h"
#include "core/cuda_array_kernels.h"

#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_cuda_sparse_tensor.h"

DipoleHamiltonianCUDASparseTensor::DipoleHamiltonianCUDASparseTensor(const libconfig::Setting &settings)
: HamiltonianStrategy(settings),
    use_double_precision(false)   // default to float precision
 {
    using std::pow;
    double r_abs;
    jblib::Vec3<double> r_ij, r_hat, s_j;

    jblib::Vec3<int> L_max(0, 0, 0);
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice.size(n));
    }

    r_cutoff_ = *std::max_element(super_cell_dim.begin(), super_cell_dim.end());

    if (settings.exists("r_cutoff")) {
        r_cutoff_ = settings["r_cutoff"];
    }


    // printf("  super cell max extent (cartesian):\n    %f %f %f\n", super_cell_dim[0], super_cell_dim[1], super_cell_dim[2]);

    for (int n = 0; n < 3; ++n) {
        if (lattice.is_periodic(n)) {
            L_max[n] = ceil(r_cutoff_/super_cell_dim[n]);
        }
    }

    printf("  image vector max extent (fractional):\n    %d %d %d\n", L_max[0], L_max[1], L_max[2]);

    dev_float_spins_.resize(globals::num_spins3);
    dev_float_fields_.resize(globals::num_spins3);

    interaction_matrix_.resize(globals::num_spins3, globals::num_spins3);

    interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    interaction_matrix_.setMatrixMode(SPARSE_FILL_MODE_LOWER);
    
    const double prefactor = kVacuumPermeadbility*kBohrMagneton/(4*kPi*pow(::lattice.parameter(),3));

    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );


    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < i; ++j) {

            if (j == i) continue;

            auto r_ij = lattice.displacement(i, j);

            const auto r_abs_sq = r_ij.norm_sq();

            if (r_abs_sq > (r_cutoff_*r_cutoff_)) continue;

        const auto r_abs = sqrt(r_abs_sq);

        const auto w0 = prefactor * globals::mus(j) / (r_abs_sq * r_abs_sq * r_abs);

        const jblib::Vec3<double> s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};
        
        const auto s_j_dot_rhat = 3.0 * dot(s_j, r_ij);
	
        r_hat = r_ij / r_abs;

        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                if (3 * i + m >= 3 * j + n) {
                    double value = (3*r_hat[m]*r_hat[n] - Id[m][n])*prefactor*globals::mus(i)*globals::mus(j)/(r_abs * r_abs * r_abs);
                                    interaction_matrix_.insertValue(3 * i + m, 3 * j + n, float(value));
              	}
            }
        }
      }

    }

    ::output.write("    converting interaction matrix format from MAP to CSR\n");
    interaction_matrix_.convertMAP2CSR();
    ::output.write("    exchange matrix memory (CSR): %f MB\n", interaction_matrix_.calculateMemory());

    // set up things on the device
    if (solver->is_cuda_solver()) { 
        cudaStreamCreate(&dev_stream_);


        cusparseStatus_t cusparse_return_status;


        ::output.write("    initialising CUSPARSE\n");
        cusparse_return_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_return_status != CUSPARSE_STATUS_SUCCESS) {
          jams_error("CUSPARSE Library initialization failed");
        }
        cusparseSetStream(cusparse_handle_, dev_stream_);


        cusparse_return_status = cusparseCreateMatDescr(&cusparse_descra_);
        if (cusparse_return_status != CUSPARSE_STATUS_SUCCESS) {
          jams_error("CUSPARSE Matrix descriptor initialization failed");
        }

        cusparseSetMatType(cusparse_descra_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
        cusparseSetMatIndexBase(cusparse_descra_, CUSPARSE_INDEX_BASE_ZERO);

        // row
        ::output.write("    allocating csr row on device\n");
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.row, (interaction_matrix_.rows()+1)*sizeof(int)));
        
        ::output.write("    memcpy csr row to device\n");
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.row, interaction_matrix_.rowPtr(),
              (interaction_matrix_.rows()+1)*sizeof(int), cudaMemcpyHostToDevice));

        // col
        ::output.write("    allocating csr col on device\n");
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.col, (interaction_matrix_.nonZero())*sizeof(int)));

        ::output.write("    memcpy csr col to device\n");
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.col, interaction_matrix_.colPtr(),
              (interaction_matrix_.nonZero())*sizeof(int), cudaMemcpyHostToDevice));

        // val
        ::output.write("    allocating csr val on device\n");
        cuda_api_error_check(
          cudaMalloc((void**)&dev_csr_interaction_matrix_.val, (interaction_matrix_.nonZero())*sizeof(float)));

        ::output.write("    memcpy csr val to device\n");
        cuda_api_error_check(cudaMemcpy(dev_csr_interaction_matrix_.val, interaction_matrix_.valPtr(),
              (interaction_matrix_.nonZero())*sizeof(float), cudaMemcpyHostToDevice));

    }
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCUDASparseTensor::calculate_total_energy() {
   double e_total = 0.0;
   for (int i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_one_spin_energy(i);
   }
    return 0.5*e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianCUDASparseTensor::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCUDASparseTensor::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCUDASparseTensor::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianCUDASparseTensor::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianCUDASparseTensor::calculate_one_spin_field(const int i, double h[3]) {
    jams_error("DipoleHamiltonianCUDASparseTensor::calculate_one_spin_field CPU unimplemented");
}


// --------------------------------------------------------------------------

void DipoleHamiltonianCUDASparseTensor::calculate_fields(jblib::Array<double, 2>& fields) {
    jams_error("DipoleHamiltonianCUDASparseTensor::calculate_fields CPU unimplemented");
}

void DipoleHamiltonianCUDASparseTensor::calculate_fields(jblib::CudaArray<double, 1>& fields) {

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
