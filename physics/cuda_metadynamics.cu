// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/cuda_metadynamics.h"

#include <libconfig.h++>

#include "core/globals.h"
#include "core/exception.h"


CudaMetadynamicsPhysics::CudaMetadynamicsPhysics(const libconfig::Setting &settings)
  : Physics(settings),
    debug_(false),
    dev_stream_(),
    dev_field_(globals::num_spins3),
    cv_theta(0),
    cv_phi(0),
    collective_variable_deriv(globals::num_spins, 3),
    gaussian_centers(),
    gaussian_width(0.3),
    gaussian_height(0.1),
    gaussian_placement_interval(100)
  {

  output.write("  * CUDA metadynamics physics module\n");

  config.lookupValue("debug", debug_);

  if (debug_) {
    ::output.write("    DEBUG ON\n");  
  }

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess){
    jams_error("Failed to create CUDA stream in CudaMetadynamicsPhysics");
  }

  // zero the field array
  if (cudaMemsetAsync(dev_field_.data(), 0.0, globals::num_spins3*sizeof(double), dev_stream_) != cudaSuccess) {
    throw cuda_api_exception("", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }

}

CudaMetadynamicsPhysics::~CudaMetadynamicsPhysics() {
}

void CudaMetadynamicsPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;

  // calculate collective variables
  calculate_collective_variables();

  if (iterations % gaussian_placement_interval == 0) {
    gaussian_centers.push_back({cv_theta, cv_phi});
    gaussian_centers.push_back({cv_theta, -cv_phi});

    // if (cv_theta - 2.0*gaussian_width < 0.0) {
    //   gaussian_centers.push_back(-cv_theta);
    // } 

    // if (cv_theta + 2.0*gaussian_width > kPi) {
    //   gaussian_centers.push_back(kPi + (kPi - cv_theta) );
    // }
    
    output_gaussians(std::cerr);
  }

  calculate_fields();

  dev_field_.copy_from_host_array(field_);
}

void CudaMetadynamicsPhysics::calculate_collective_variables() {

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
    collective_variable_deriv(i, 2) = m0 * (mz / mm) * globals::s(i, 2) - mm;

    // phi
    collective_variable_deriv(i, 0) += mp * (-my * globals::s(i, 0));
    collective_variable_deriv(i, 1) += mp * (mx * globals::s(i, 1));
    collective_variable_deriv(i, 2) += 0.0;
  }
}

void CudaMetadynamicsPhysics::calculate_fields() {

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

//  for (auto i = 0; i < globals::num_spins; ++i) {
//      std::cerr << field_(i, 0)<< "\t" << field_(i, 1) << "\t" << field_(i, 2) << std::endl;
//  }
//  exit(0);

}


double CudaMetadynamicsPhysics::gaussian(double x, double y) {
  return gaussian_height * exp(- ((0.5 * x * x / (gaussian_width * gaussian_width))
                                 +(0.5 * y * y / (gaussian_width * gaussian_width))));
}

void CudaMetadynamicsPhysics::output_gaussians(std::ostream &out) {
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