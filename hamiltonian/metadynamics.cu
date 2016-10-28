#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"

#include "hamiltonian/metadynamics.h"

MetadynamicsHamiltonian::MetadynamicsHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings),
  cv_theta(0),
  cv_phi(0),
  collective_variable_deriv(globals::num_spins, 3),
  gaussian_centers(),
  gaussian_width(0.2),
  gaussian_height(1.0),
  gaussian_placement_interval(5000)
{
    ::output.write("initialising metadynamics Hamiltonian\n");
    // output in default format for now
    outformat_ = TEXT;

    // resize member arrays
    energy_.resize(globals::num_spins);
    energy_.zero();
    field_.resize(globals::num_spins, 3);
    field_.zero();


    // transfer arrays to cuda device if needed
// #ifdef CUDA
//     if (solver->is_cuda_solver()) {
//         cudaStreamCreate(&dev_stream_);

//         dev_energy_ = jblib::CudaArray<double, 1>(energy_);
//         dev_field_  = jblib::CudaArray<double, 1>(field_);
//     }
// #endif

}

// --------------------------------------------------------------------------


void MetadynamicsHamiltonian::calculate_collective_variables() {
    // DO THIS ON THE GPU EVENTUALLY
    Vec3 mag = {0.0, 0.0, 0.0};

    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        mag[j] += globals::s(i, j);
      }
    }

    cv_theta = azimuthal_angle(mag);
    cv_phi = polar_angle(mag);

    const auto mm = abs(mag);
    const auto mx = mag.x;
    const auto my = mag.y;
    const auto mz = mag.z;


     auto mp = 1.0/(mx*mx + my*my);
    if (isinf(mp)) {
      mp = 1e100;
    }

    auto m0 = 1.0 / (mm * mm * sqrt(1.0 - (mz * mz) / (mm * mm) ));
    //auto m0 = 1.0;
    if (isinf(m0)) {
      m0 = 1e100;
    }

    for (int i = 0; i < globals::num_spins; ++i) {
      // theta
      collective_variable_deriv(i, 0) = m0 * (mz / mm) * globals::s(i, 0);
      collective_variable_deriv(i, 1) = m0 * (mz / mm) * globals::s(i, 1);
      collective_variable_deriv(i, 2) = m0 * ((mz / mm) * globals::s(i, 2) - mm);

      // phi
      collective_variable_deriv(i, 0) += mp * (-my * globals::s(i, 0));
      collective_variable_deriv(i, 1) += mp * (mx * globals::s(i, 1));
      collective_variable_deriv(i, 2) += 0.0;
    }
}

double MetadynamicsHamiltonian::gaussian(double x, double y) {
  return gaussian_height * exp(- ((0.5 * x * x / (gaussian_width * gaussian_width))
                                  +(0.5 * y * y / (gaussian_width * gaussian_width))
                                 ));
}

void MetadynamicsHamiltonian::output_gaussians(std::ostream &out) {
  auto theta = 0.0;
  auto delta_theta = gaussian_width/5.0;
  auto delta_phi = gaussian_width/5.0;

  do {
    auto phi = -kPi;
    do {
      auto potential = 0.0;

      for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){
        auto x = (theta - (*it)[0]);
        auto y = (phi - (*it)[1]);
        potential += gaussian(x, y);
      }

      std::cerr << theta << "\t" << phi << "\t" << potential << std::endl;

      phi +=delta_phi;
    } while (phi < kPi);

    theta += delta_theta;
  } while (theta < kPi);

  std::cerr << "\n\n";

}

void MetadynamicsHamiltonian::add_gaussian() {
    calculate_collective_variables();
    gaussian_centers.push_back({cv_theta, cv_phi});
    // gaussian_centers.push_back({cv_theta, -cv_phi});
    // gaussian_centers.push_back({kPi-cv_theta, cv_phi});
    // gaussian_centers.push_back({kPi-cv_theta, -cv_phi});

    // if(cv_theta < 2.0*gaussian_width) {
      // gaussian_centers.push_back({-cv_theta, cv_phi});
      // gaussian_centers.push_back({-cv_theta, -cv_phi});
      // gaussian_centers.push_back({kPi+cv_theta, cv_phi});
      // gaussian_centers.push_back({kPi+cv_theta, -cv_phi});
    // }
}

double MetadynamicsHamiltonian::calculate_total_energy() {
    calculate_collective_variables();

    auto potential = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      auto x = (cv_theta - (*it)[0]);
      auto y = (cv_phi - (*it)[1]) ;

      potential += gaussian(x, y);
    }

    return potential;
}

// --------------------------------------------------------------------------

double MetadynamicsHamiltonian::calculate_one_spin_energy(const int i) {
  return calculate_total_energy();
}

// --------------------------------------------------------------------------

double MetadynamicsHamiltonian::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    Vec3 mag = {0.0, 0.0, 0.0};

    for (auto n = 0; n < globals::num_spins; ++n) {
      if (n == i) {
        continue;
      }

      for (auto j = 0; j < 3; ++j) {
        mag[j] += globals::s(n, j);
      }
    }

    cv_theta = azimuthal_angle(mag + spin_initial);
    cv_phi = polar_angle(mag + spin_initial);

    auto e_initial = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      auto x = (cv_theta - (*it)[0]);
      auto y = (cv_phi - (*it)[1]) ;

      e_initial += gaussian(x, y);
    }


    cv_theta = azimuthal_angle(mag + spin_final);
    cv_phi = polar_angle(mag + spin_final);

    auto e_final = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      auto x = (cv_theta - (*it)[0]);
      auto y = (cv_phi - (*it)[1]) ;

      e_final += gaussian(x, y);
    }

    return  e_final-e_initial;
}

// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {

}



// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::calculate_fields() {
  calculate_collective_variables();

  auto potential_deriv = 0.0;
  for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

    auto x = (cv_theta - (*it)[0]);
    auto y = (cv_phi - (*it)[1]) ;


    potential_deriv = potential_deriv - 0.5* gaussian(x, y) * x / (gaussian_width * gaussian_width)
                                      - 0.5* gaussian(x, y) * y / (gaussian_width * gaussian_width);
  }

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      field_(i, j) = potential_deriv * collective_variable_deriv(i, j);
    }
  }

  // std::cout << field_(0, 0) << "\t" << field_(0, 1) << "\t" << field_(0, 2) << std::endl;
}
// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::output_energies(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_energies_text();
        case HDF5:
            jams_error("metadynamics energy output: HDF5 not yet implemented");
        default:
            jams_error("metadynamics energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::output_fields(OutputFormat format) {
    switch(format) {
        case TEXT:
            output_fields_text();
        case HDF5:
            jams_error("metadynamics energy output: HDF5 not yet implemented");
        default:
            jams_error("metadynamics energy output: unknown format");
    }
}

// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::output_energies_text() {

}

// --------------------------------------------------------------------------

void MetadynamicsHamiltonian::output_fields_text() {

}

double MetadynamicsHamiltonian::calculate_bond_energy_difference(const int i, const int j, const Vec3 &sj_initial, const Vec3 &sj_final) {
  if (i != j) {
    return 0.0;
    } else {
  return calculate_one_spin_energy_difference(i, sj_initial, sj_final);
    }
}
