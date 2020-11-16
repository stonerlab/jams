#include <jams/metadynamics/magnetisation_cv.h>

#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>

#include <fstream>

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

jams::MagnetisationCollectiveVariable::MagnetisationCollectiveVariable() {
  gaussian_amplitude_ = 1e-25;
  gaussian_width_ = 0.02;

  sample_points_ = linear_space(-2.0, 2.0, 0.01);
  potential_.resize(sample_points_.size(), 0.0);

  magnetisation_ = calculate_total_magnetisation();
}

jams::MagnetisationCollectiveVariable::MagnetisationCollectiveVariable(
    const libconfig::Setting &settings) : MagnetisationCollectiveVariable() {}


void jams::MagnetisationCollectiveVariable::insert_gaussian(
    const double &relative_amplitude) {
  assert(sample_points_.size() == potential_.size());

  auto center = collective_coordinate();

  for (auto i = 0; i < sample_points_.size(); ++i) {
    potential_[i] += gaussian(
        sample_points_[i], center, gaussian_amplitude_, gaussian_width_);
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
  for (auto i = 0; i < sample_points_.size(); ++i) {
    potential_[i] += relative_amplitude * gaussian(sample_points_[i], mirrored_center, gaussian_amplitude_, gaussian_width_);
  }

  // recalculate total magnetisation to avoid numerical drift
  magnetisation_ = calculate_total_magnetisation();
}

void jams::MagnetisationCollectiveVariable::output(
    std::ofstream &of) {
  assert(of.isopen());
  for (auto i = 0; i < sample_points_.size(); ++i) {
    of << i << " " << sample_points_[i] << " " << potential_[i] << "\n";
  }
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

double jams::MagnetisationCollectiveVariable::interpolated_potential(
    const double &value) {
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
