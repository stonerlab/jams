// init_skyrmion.cc                                                    -*-C++-*-

#include <jams/initializer/init_skyrmion.h>
#include <jams/interface/config.h>
#include <jams/core/globals.h>
#include <jams/core/lattice.h>

#include <cmath>

void jams::InitSkyrmion::execute(const libconfig::Setting &settings) {
  //
  auto is_afm = jams::config_required<int>(settings, "is_afm");

  double w = jams::config_optional<double>(settings, "w", 5.0);
  double c = jams::config_optional<double>(settings, "c", 5.0);

  auto coordinate_format = jams::config_optional<CoordinateFormat>(settings, "coordinate_format", CoordinateFormat::FRACTIONAL);

  // We need to use some extra code below to ensure that the default center (if
  // none is specified in the config) for both fractional and cartesian
  // coordinates is the same (the center of the x-y plane).
  Vec2 center;
  switch (coordinate_format) {
    case CoordinateFormat::FRACTIONAL: {
      center = jams::config_optional<Vec2>(settings, "center", Vec2{0.5, 0.5});
      auto frac_center = globals::lattice->fractional_to_cartesian(
          {center[0] * globals::lattice->size(0), center[1] * globals::lattice->size(1),
           center[1] * globals::lattice->size(1)});
      center = Vec2{frac_center[0], frac_center[1]};
      }
      break;
    case CoordinateFormat::CARTESIAN: {
      auto frac_center = globals::lattice->fractional_to_cartesian(
          {0.5 * globals::lattice->size(0), 0.5 * globals::lattice->size(1), 0.0});
      center = jams::config_optional<Vec2>(settings, "center",
                                           Vec2{frac_center[0],
                                                frac_center[1]});
      }
      break;
  }


  // Defaults to a Bloch skyrmion with a negative charge (core pointing
  // in -z direction within a +z FM)
  double Q = jams::config_optional<double>(settings, "polarity", -1);
  // ensure Q is either +1 or -1 (if it's zero then we simply have no skyrmion)
  Q = Q / std::abs(Q);

  double Qv = jams::config_optional<double>(settings, "vorticity", 1.0);
  double Qh = jams::config_optional<double>(settings, "helicity", 0.0);

  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r_i = globals::lattice->displacement({center[0], center[1], 0.0}, globals::lattice->atom_position(i));
    double x = r_i[0];
    double y = r_i[1];
    double r = sqrt(x*x + y*y);

    // calculate orientation of skyrmion spin in perfect (T=0K) state

    // https://juspin.de/skyrmion-radius/
    // https://iopscience.iop.org/article/10.1088/1361-648X/ab5488
    double theta = asin(std::tanh(-(r+c)/(w/2.0))) + asin(std::tanh(-(r-c)/(w/2.0))) + kPi;
    double phi = std::atan2(y, x);

    // if we're dealing with an antiferromagnet, orient the spins composing the skyrmion accordingly
    if (is_afm) {
      globals::lattice
      if (globals::lattice->atom_material_id(i) == 0) {
        // by convention, we'll hardcode this material to have the skyrmion core pointing along -z.
        globals::s(i, 0) = -Q * sin(theta) * cos(Qv*phi + Qh);
        globals::s(i, 1) = -Q * sin(theta) * sin(Qv*phi + Qh);
        globals::s(i, 2) = -Q * cos(theta);
      } else {
        globals::s(i, 0) = Q * sin(theta) * cos(Qv*phi + Qh);
        globals::s(i, 1) = Q * sin(theta) * sin(Qv*phi + Qh);
        globals::s(i, 2) = Q * cos(theta);
      }
    } else {
      globals::s(i, 0) = -Q * sin(theta) * cos(Qv*phi + Qh);
      globals::s(i, 1) = -Q * sin(theta) * sin(Qv*phi + Qh);
      globals::s(i, 2) = -Q * cos(theta);
    }

  }
}

// ----------------------------- END-OF-FILE ----------------------------------
