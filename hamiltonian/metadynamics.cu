#include "core/globals.h"
#include "core/utils.h"
#include "core/maths.h"
#include "core/consts.h"
#include "core/cuda_defs.h"

#include "hamiltonian/metadynamics.h"

MetadynamicsHamiltonian::MetadynamicsHamiltonian(const libconfig::Setting &settings)
: Hamiltonian(settings),
  // cv_mag_x(0),
  // cv_mag_y(0),
  // cv_mag_z(0),
  cv_mag_t(0),
  cv_mag_p(0),
  collective_variable_deriv(globals::num_spins, 3),
  gaussian_centers(),
  gaussian_width(0.05),
  gaussian_height(0.1),
  gaussian_placement_interval(100)
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

    mag /= double(globals::num_spins);

    // cv_mag_x = mag.x;
    // cv_mag_y = mag.y;
    // cv_mag_z = mag.z;

    cv_mag_t = sqrt(mag.x * mag.x + mag.y * mag.y);
    cv_mag_p = mag.z;

//     const auto mm = abs(mag);
//     const auto mx = mag.x;
//     const auto my = mag.y;
//     const auto mz = mag.z;


//      auto mp = 1.0/(mx*mx + my*my);
//     if (isinf(mp)) {
//       mp = 1e100;
//     }

//     auto m0 = 1.0 / (mm * mm * sqrt(1.0 - (mz * mz) / (mm * mm) ));
//     //auto m0 = 1.0;
//     if (isinf(m0)) {
//       m0 = 1e100;
//     }

//     for (int i = 0; i < globals::num_spins; ++i) {
//       // theta
//       collective_variable_deriv(i, 0) = m0 * (mz / mm) * globals::s(i, 0);
//       collective_variable_deriv(i, 1) = m0 * (mz / mm) * globals::s(i, 1);
//       collective_variable_deriv(i, 2) = m0 * ((mz / mm) * globals::s(i, 2) - mm);

//       // phi
//       collective_variable_deriv(i, 0) += mp * (-my * globals::s(i, 0));
//       collective_variable_deriv(i, 1) += mp * (mx * globals::s(i, 1));
//       collective_variable_deriv(i, 2) += 0.0;
//     }
}

double MetadynamicsHamiltonian::gaussian(double x, double y) {
  // return gaussian_height * exp(- ((0.5 * x * x / (gaussian_width * gaussian_width))
  //                                 +(0.5 * y * y / (gaussian_width * gaussian_width))
  //                                 +(0.5 * z * z / (gaussian_width * gaussian_width))
  //                                ));
  return gaussian_height * exp(- ((0.5 * x * x / (gaussian_width * gaussian_width))
                                  +(0.5 * y * y / (gaussian_width * gaussian_width))
                                 ));
}

void MetadynamicsHamiltonian::output_gaussians(std::ostream &out) {
  auto delta_t = gaussian_width/3.0;
  auto delta_p = gaussian_width/3.0;

  auto mt = 0.0;
  do {
    auto mp = -1.0;
    do {
      auto potential = 0.0;

      for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){
        auto x = (mt - (*it)[0]);
        auto y = (mp - (*it)[1]);
        potential += gaussian(x, y);
      }

      std::cerr << mt << "\t" << mp << "\t" << potential << std::endl;

      mp +=delta_p;
    } while (mp < 1.0);

    mt += delta_t;
  } while (mt < 1.0);

  std::cerr << "\n\n";

}

void MetadynamicsHamiltonian::add_gaussian() {
    calculate_collective_variables();
    gaussian_centers.push_back({cv_mag_t, cv_mag_p});
    // gaussian_centers.push_back({cv_mag_x, cv_mag_y, cv_mag_z});

    // gaussian_centers.push_back({cv_mag_para, -cv_mag_perp});
    // gaussian_centers.push_back({kPi-cv_mag_para, cv_mag_perp});
    // gaussian_centers.push_back({kPi-cv_mag_para, -cv_mag_perp});

    // if(cv_mag_para < 2.0*gaussian_width) {
      // gaussian_centers.push_back({-cv_mag_para, cv_mag_perp});
      // gaussian_centers.push_back({-cv_mag_para, -cv_mag_perp});
      // gaussian_centers.push_back({kPi+cv_mag_para, cv_mag_perp});
      // gaussian_centers.push_back({kPi+cv_mag_para, -cv_mag_perp});
    // }
}

double MetadynamicsHamiltonian::calculate_total_energy() {
    calculate_collective_variables();

    auto potential = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      // auto x = (cv_mag_x - (*it)[0]);
      // auto y = (cv_mag_y - (*it)[1]);
      // auto z = (cv_mag_z - (*it)[2]);

      auto x = (cv_mag_t - (*it)[0]);
      auto y = (cv_mag_p - (*it)[1]);

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

    auto mag_initial = (mag + spin_initial) / double(globals::num_spins);

    // cv_mag_x = mag_initial.x;
    // cv_mag_y = mag_initial.y;
    // cv_mag_z = mag_initial.z;

    cv_mag_t = sqrt(mag_initial.x * mag_initial.x + mag_initial.y * mag_initial.y);
    cv_mag_p = mag_initial.z;

    auto e_initial = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      auto x = (cv_mag_t - (*it)[0]);
      auto y = (cv_mag_p - (*it)[1]);
      // auto z = (cv_mag_z - (*it)[2]);

      e_initial += gaussian(x, y);
    }


    auto mag_final = (mag + spin_initial) / double(globals::num_spins);

    // cv_mag_x = mag_final.x;
    // cv_mag_y = mag_final.y;
    // cv_mag_z = mag_final.z;

    cv_mag_t = sqrt(mag_final.x * mag_final.x + mag_final.y * mag_final.y);
    cv_mag_p = mag_final.z;

    auto e_final = 0.0;
    for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

      // auto x = (cv_mag_x - (*it)[0]);
      // auto y = (cv_mag_y - (*it)[1]);
      // auto z = (cv_mag_z - (*it)[2]);

      auto x = (cv_mag_t - (*it)[0]);
      auto y = (cv_mag_p - (*it)[1]);

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
  // calculate_collective_variables();

  // auto potential_deriv = 0.0;
  // for (auto it = gaussian_centers.begin(); it != gaussian_centers.end(); ++it){

  //   auto x = (cv_mag_para - (*it)[0]);
  //   auto y = (cv_mag_perp - (*it)[1]) ;


  //   potential_deriv = potential_deriv - 0.5* gaussian(x, y) * x / (gaussian_width * gaussian_width)
  //                                     - 0.5* gaussian(x, y) * y / (gaussian_width * gaussian_width);
  // }

  // for (auto i = 0; i < globals::num_spins; ++i) {
  //   for (auto j = 0; j < 3; ++j) {
  //     field_(i, j) = potential_deriv * collective_variable_deriv(i, j);
  //   }
  // }

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
