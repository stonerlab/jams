// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_METADYNAMICS_H
#define JAMS_PHYSICS_METADYNAMICS_H

#include <libconfig.h++>

#include <vector>

#include "core/physics.h"
#include "jblib/containers/cuda_array.h"


class CudaMetadynamicsPhysics : public Physics {
 public:
  CudaMetadynamicsPhysics(const libconfig::Setting &settings);
  ~CudaMetadynamicsPhysics();
  void update(const int &iterations, const double &time, const double &dt);

 // override the base class implementation
 const double* field() { return dev_field_.data(); }

private:

	void calculate_collective_variables();
	void calculate_fields();
	void output_gaussians(std::ostream &out);

   bool debug_;

   cudaStream_t                dev_stream_;

   jblib::CudaArray<double, 1> dev_field_;

   double			   	   cv_theta;
   double			   	   cv_phi;
   jblib::Array<double, 2> collective_variable_deriv;

   std::vector<double> gaussian_centers;   
   double 			   gaussian_width;
   double 			   gaussian_height;
   int                 gaussian_placement_interval;
};

#endif  // JAMS_PHYSICS_METADYNAMICS_H
