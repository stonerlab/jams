#include <jams/metadynamics/magnetisation_cv.h>

#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>
#include <libconfig.h++>
#include <fstream>
#include <jams/interface/config.h>
#include "jams/helpers/output.h"
#include "jams/core/solver.h"
#include <string>

namespace {
    std::vector<double> linear_space(const double &min,const double &max,const double &step) {
      assert(min < max);
      std::vector<double> space;
      double value = min;
      while (value < max+step) {
        space.push_back(value);
        value += step;
      }

      return space;
    }
}

jams::MagnetisationCollectiveVariable::MagnetisationCollectiveVariable(const libconfig::Setting &settings) {

  // ---------------------------------------------------------------------------
  // config settings
  // ---------------------------------------------------------------------------

  // maximum amplitude of inserted gaussians in Joules
  // (this can be reduced by tempering in the metadynamics solver)
  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude" ) / kBohrMagneton;

  // width of the gaussian in units of mz
  gaussian_width_ = jams::config_required<double>(settings, "gaussian_width") ;

  // discretisation width of the metadynamics potential landscape in units of mz
  histogram_step_size_ = jams::config_required<double>(settings,"histogram_step_size");

  metadynamics_simulation_parameters.open(jams::output::full_path_filename(sim_type_selected+"_parameters.tsv"));
  metadynamics_simulation_parameters << "iterations" << "	" << "gaussian_amplitude" << "	" << "energy_barrier" <<"\n";
  potential.open(jams::output::full_path_filename(sim_type_selected+"_potential.tsv"));
  potential << "N(s(x),t)" << "	" << "V(s(x),t)" <<"\n";


  sample_points_ = linear_space(-2.0, 2.0, histogram_step_size_);
  potential_.resize(sample_points_.size(), 0.0);
  physical_region_indices();


  magnetisation_ = calculate_total_magnetisation();
}

void jams::MagnetisationCollectiveVariable::insert_gaussian(const double &relative_amplitude) {
  assert(sample_points_.size() == potential_.size());

  auto center = collective_coordinate();

  for (auto i = 0; i < sample_points_.size(); ++i) {
    potential_[i] += relative_amplitude * gaussian(sample_points_[i],
                                                   center, gaussian_amplitude_, gaussian_width_);
  }
  // calculate the center position for a gaussian according to mirror boundary conditions
  double mirrored_center;
  if (center >=0) {
    mirrored_center = 2 - center;
  } else {
    mirrored_center = -2 - center;
  }
  assert(mirrored_center >= -2 && mirrored_center <= 2);

  // insert the mirrored gaussian
  for (auto i = 0; i < potential_.size(); ++i) {
    potential_[i] += relative_amplitude * gaussian(sample_points_[i],
                                                   mirrored_center, gaussian_amplitude_, gaussian_width_);
  }

  // recalculate total magnetisation to avoid numerical drift
  magnetisation_ = calculate_total_magnetisation();
}


void jams::MagnetisationCollectiveVariable::output() {
    potential.open(jams::output::full_path_filename(sim_type_selected+"_potential.tsv"));
    potential << "N(s(x),t)" << "	" << "V(s(x),t)" <<"\n";
     for (auto i = lower_limit_index; i < upper_limit_index +1; ++i) {
	   potential << sample_points_[i] <<"	"<< potential_[i] * kBohrMagneton <<  "\n";
      }
    potential.close();
    metadynamics_simulation_parameters <<solver->iteration() <<"	"<< gaussian_amplitude_used << "	"<<histogram_energy_difference() << "\n";
}

double jams::MagnetisationCollectiveVariable::current_potential() {
  return interpolated_potential(magnetisation_[2] / globals::num_spins);
}

double jams::MagnetisationCollectiveVariable::potential_difference(int i,
                                                                   const Vec3 &spin_initial,
                                                                   const Vec3 &spin_final) {
  const Vec3 initial_magnetisation = magnetisation_;
  auto initial_potential = interpolated_potential(initial_magnetisation[2] / globals::num_spins);

  const Vec3 trial_magnetisation = magnetisation_ - spin_initial + spin_final;
  auto trial_potential = interpolated_potential(trial_magnetisation[2] / globals::num_spins);

  return trial_potential - initial_potential;
}

Vec3 jams::MagnetisationCollectiveVariable::calculate_total_magnetisation() {
  Vec3 m={0, 0, 0};
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto n = 0; n < 3; ++n) {
      m[n] += globals::s(i, n);
    }
  }
  return m;
}

double jams::MagnetisationCollectiveVariable::interpolated_potential(const double &value) {
  assert(is_sorted(begin(sample_points_), end(sample_points_)));
  assert(value > sample_points_.front() || approximately_equal(sample_points_.front(), value));
  assert(value < sample_points_.back() || approximately_equal(sample_points_.back(), value));
  assert(sample_points_.size() == potential_.size());
  // TODO: Test if this gives the correct points for the interpolation
  auto lower = floor((value - sample_points_[0]) / (sample_points_[1] - sample_points_[0]));
  auto upper = lower+1;
  assert(lower < upper);
  //cout << "Indices Lower:" << lower <<endl; //need to check why why and why
  return jams::maths::linear_interpolation(value,
                                           sample_points_[lower], potential_[lower],
                                           sample_points_[upper], potential_[upper]);
}

double jams::MagnetisationCollectiveVariable::collective_coordinate() {
  return magnetisation_[2] / globals::num_spins;
}

void jams::MagnetisationCollectiveVariable::spin_update(int i,
                                                        const Vec3 &spin_initial,
                                                        const Vec3 &spin_final) {
  magnetisation_ = magnetisation_ - spin_initial + spin_final;
}

 double jams::MagnetisationCollectiveVariable::histogram_energy_difference() {
  const auto margin = potential_.size()/4;
  const double max = *max_element(potential_.begin()+margin, potential_.end() -margin);
  const double min = *min_element(potential_.begin()+margin, potential_.end() -margin);
  return max - min;
}
void jams::MagnetisationCollectiveVariable::physical_region_indices() {
  auto lower_limit = -1;
  auto upper_limit = 1;
  for (auto i = 0; i < sample_points_.size(); ++i ) {
	if (approximately_equal(sample_points_[i],double(lower_limit))) {
	  lower_limit_index= i;
	  assert(sample_points_[i]<=lower_limit_index);
	  break;
	}}
  for (auto ii=lower_limit_index; ii<sample_points_.size(); ++ii){
	if ( approximately_equal(double(upper_limit), sample_points_[ii])) {
	  upper_limit_index = ii;
	  assert(sample_points_[ii]<=upper_limit_index);
	  break;
	}}


}
