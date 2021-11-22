//// skyrmion_center_cv.h                                                -*-C++-*-
//#ifndef INCLUDED_JAMS_METADYNAMICS_SKYRMION_CENTER_CV
//#define INCLUDED_JAMS_METADYNAMICS_SKYRMION_CENTER_CV
//
//#include "jams/metadynamics/metadynamics_potential.h"
//
//#include "jams/interface/config.h"
//#include "jams/containers/multiarray.h"
//#include "jams/containers/vec3.h"
//
//#include <fstream>
//#include <vector>
//#include <string>
//
//namespace jams {
//class SkyrmionCenterCV : public MetadynamicsPotential {
//private:
//    double gaussian_amplitude_;
//    double gaussian_width_;
//
//    double histogram_step_size_;
//
//    /// Threshold value of s_z for considering a spin as part of the core of
//    /// the skyrmion for the purposes of calculating the centre.
//    double skyrmion_core_threshold_;
//
//    std::vector<double> cv_samples_x_;
//    std::vector<double> cv_samples_y_;
//
//    jams::MultiArray<double, 2> potential_;
////    std::vector<std::vector<double>> potential_;
//
//    // The constructor for this class is usually called before the spin
//    // configuration is finalised so the centre of mass will be incorrect if the
//    // cache is initialised inside of the constructor. We therefore us this flag
//    // as a hack for the first setup of the cache. It may be better to eventually
//    // rewrite this with a "proper" caching system around the centre of mass
//    // function.
//    bool do_first_cache_ = true;
//
//    bool periodic_x_ = true;
//    bool periodic_y_ = true;
//
//    Vec3 cached_initial_center_of_mass_ = {0.0, 0.0, 0.0};
//    Vec3 cached_trial_center_of_mass_ = {0.0, 0.0, 0.0};
//
//    std::vector<Vec3> cylinder_remapping_x_;
//    std::vector<Vec3> cylinder_remapping_y_;
//
//    std::ofstream potential_landscape;
//    std::ofstream skyrmion_outfile;
//    std::ofstream skyrmion_com; //track the center of mass location, plot its path in python.
//
// public:
//  SkyrmionCenterCV(const libconfig::Setting &settings);
//
//  void insert_gaussian(const double& relative_amplitude) override;
//
//  void output() override;
//
//  double potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;
//
//  double current_potential() override;
//
//  void spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;
//
// private:
//    /// Maps the 2D x-y plane onto two cylinder coordinate systems to allow
//    /// calculation of the center of mass with periodic boundaries. The remapped
//    /// coordinates on the cylinders are stored in cylinder_remapping_x_ and
//    /// cylinder_remapping_y_. Because the lattice is fixed this only needs to
//    /// be done once on initialisation.
//    void space_remapping();
//
//
//    void output_remapping();
//
//    /// Returns true if the z component of spin crosses the given threshold when
//    /// changing from s_initial to s_final.
//    bool spin_crossed_threshold(const Vec3& s_initial, const Vec3& s_final, const double& threshold);
//
//    /// Returns the center of mass of the skyrmion. This is defined as the
//    /// average of all points where s_z < skyrmion_core_threshold_. Periodic
//    /// boundaries (if they are turned on) are account for by the space
//    /// remapping which is precalculated at initialisation.
//    Vec3 skyrmion_center_of_mass();
//
//    double interpolated_potential(const double& x, const double& y);
//
//    void import_potential(const std::string &filename);
//
//}
//;}
//
//#endif //INCLUDED_JAMS_METADYNAMICS_SKYRMION_CENTER_CV