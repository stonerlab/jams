// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_METADYNAMICS_H
#define JAMS_PHYSICS_METADYNAMICS_H

#include <libconfig.h++>

#include <vector>
#include <array>

#include "core/physics.h"
#include "hamiltonian/metadynamics.h"


class MetadynamicsPhysics : public Physics {
 public:
  MetadynamicsPhysics(const libconfig::Setting &settings);
  ~MetadynamicsPhysics();
  void update(const int &iterations, const double &time, const double &dt);

private:

	void calculate_collective_variables();
	void calculate_fields();
  void calculate_potential();

	void output_gaussians(std::ostream &out);
	double gaussian(double x, double y);

   bool debug_;

   // make shared pointer instead
   MetadynamicsHamiltonian* meta_hamiltonian;

   double			   	   cv_theta;
   double			   	   cv_phi;

   jblib::Array<double, 2> collective_variable_deriv;

   std::vector<std::array<double, 2>> gaussian_centers;   
   double 			   gaussian_width;
   double 			   gaussian_height;
   int             gaussian_placement_interval;
};

#endif  // JAMS_PHYSICS_METADYNAMICS_H
