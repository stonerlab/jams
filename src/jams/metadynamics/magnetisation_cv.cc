#include <jams/metadynamics/magnetisation_cv.h>

#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>
#include <libconfig.h++>
#include <fstream>
#include <jams/interface/config.h>
#include "jams/helpers/output.h"
#include "jams/core/solver.h"
#include <string>
#include <algorithm>

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

    int find_equal_index(const std::vector<double>& container, const double& value) {
      assert(is_sorted(begin(container), end(container)));
      auto result = std::find_if(container.begin(), container.end(),
                                   [&](double x){ return approximately_equal(x, value); });

      // failed to find the index
      assert (result != container.end());

      return result - container.begin();
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

  // ---------------------------------------------------------------------------
  // validate settings
  // ---------------------------------------------------------------------------

  // If histogram_step_size does not divide evenly into the range -1 -> 1 then
  // we will be missing either the start of the end point of the physical range.
  if (!approximately_equal(std::remainder(2.0, histogram_step_size_), 0.0)) {
    throw std::runtime_error("Invalid value of histogram_step_size: "
                             "histogram_step_size must divide into 2.0 with no remainder");
  }

  // ---------------------------------------------------------------------------

  sample_points_ = linear_space(-2.0, 2.0, histogram_step_size_);
  lower_limit_index = find_equal_index(sample_points_, -1.0);
  upper_limit_index = find_equal_index(sample_points_, 1.0);

  potential_.resize(sample_points_.size(), 0.0);

  if (settings.exists("potential_file")) {
    // TODO : finish the code test and copy it here (passing a potential file)
  }


  magnetisation_ = calculate_total_magnetisation();
}

void jams::MagnetisationCollectiveVariable::insert_gaussian(
    const double &relative_amplitude) {
void jams::MagnetisationCollectiveVariable::insert_gaussian(const double &relative_amplitude) {
  assert(sample_points_.size() == potential_.size());

  // recalculate total magnetisation to avoid numerical drift from the
  // addition and subtractions in spin_update()
  magnetisation_ = calculate_total_magnetisation();

  double cv_center = collective_variable();

  // calculate the center position for a gaussian according to mirror boundary conditions
  double mirrored_center = cv_center >= 0.0 ? 2.0 - cv_center : -2.0 - cv_center;
  assert(mirrored_center >= -2.0 && mirrored_center <= 2.0);

  // insert gaussians into the discretised potential
  for (auto i = 0; i < potential_.size(); ++i) {
    for (const auto &center : {cv_center, mirrored_center}) {
      potential_[i] += gaussian(sample_points_[i], center,
                                gaussian_amplitude_ * relative_amplitude,
                                gaussian_width_);
    }
  }
}


void jams::MagnetisationCollectiveVariable::output() {
    std::ofstream potential_output_file(jams::output::full_path_filename("potential.tsv"));
    potential_output_file << "# m_z metad_potential_joules\n";

    for (auto i = lower_limit_index; i < upper_limit_index + 1; ++i) {
      potential_output_file << sample_points_[i] << "	" << potential_[i] * kBohrMagneton << "\n";
    }
    potential_output_file.close();


    if (!potential_difference_output_file_.is_open()) {
      // open the output file to track the potential difference
      potential_difference_output_file_.open(
          jams::output::full_path_filename("potential_difference.tsv"));
      potential_difference_output_file_
          << "# iteration metad_potential_diff_joules" << "\n";
    }
    
    potential_difference_output_file_ << solver->iteration() << "	" << histogram_energy_difference()*kBohrMagneton << std::endl;
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
  // EXPENSIVE assert to catch assumption of sorted sample_points_
  //  assert(is_sorted(begin(sample_points_), end(sample_points_)));

  assert(value > sample_points_.front() || approximately_equal(sample_points_.front(), value));
  assert(value < sample_points_.back() || approximately_equal(sample_points_.back(), value));
  assert(sample_points_.size() == potential_.size());

  auto lower = floor((value - sample_points_[0]) / histogram_step_size_);
  auto upper = lower+1;
  assert(lower < upper);

  return jams::maths::linear_interpolation(value,
                                           sample_points_[lower], potential_[lower],
                                           sample_points_[upper], potential_[upper]);
}

double jams::MagnetisationCollectiveVariable::collective_variable() {
  return magnetisation_[2] / globals::num_spins;
}

void jams::MagnetisationCollectiveVariable::spin_update(int i,
                                                        const Vec3 &spin_initial,
                                                        const Vec3 &spin_final) {
  magnetisation_ = magnetisation_ - spin_initial + spin_final;
}

 double jams::MagnetisationCollectiveVariable::histogram_energy_difference() {
  // margin in the number of elements in potential in the "virtual" space
  // outside of the range [-1,1]
  const auto margin = potential_.size()/4;

  // midpoint is in the center of the potential space
  const auto midpoint = potential_.size()/2;

  // these are iterators to the maximum point between the edged (-1 or +1) and
  // the midpoint. In the free energy (negative of the potential) these are
  // the minimum energy wells in a uniaxial system
  const auto left_max_it = max_element(potential_.begin()+margin, potential_.begin() + midpoint);
  const auto right_max_it = max_element(potential_.begin()+midpoint, potential_.end() - margin);

  // this is the potential energy minimum (free energy maximum) between the
  // two maxima in a uniaxial system
  const double min_energy = *min_element(left_max_it, right_max_it);

  // we use the maximum of the two maxes for the estimated energy barrier
  const double max_energy = std::max(*left_max_it, *right_max_it);
  return min_energy - max_energy;
}
