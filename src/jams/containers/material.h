//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_MATERIAL_H
#define JAMS_MATERIAL_H

#include <string>
#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

class Material {
public:
    int           id = 0;
    std::string name = "";
    double    moment = 0.0;
    double     gyro  = jams::defaults::material_gyro;
    double     alpha = jams::defaults::material_alpha;
    jams::Vec<double, 3>       spin  = jams::defaults::material_spin;
    jams::Mat<double, 3, 3>   transform = jams::defaults::material_spin_transform;
    bool   randomize = false;

    inline Material() = default;

    inline explicit Material(const libconfig::Setting& cfg) :
            id       (0),
            name     (jams::config_required<std::string>(cfg, "name")),
            moment   (jams::config_required<double>(cfg, "moment") * kBohrMagnetonIU), // input is in multiples of Bohr magneton, internally we use meV T^-1
            gyro     (jams::config_optional<double>(cfg, "gyro", jams::defaults::material_gyro) * kGyromagneticRatioIU), // input is fraction of the gyromagnetic ratio, internally we use rad ps^-1 T^-1
            alpha    (jams::config_optional<double>(cfg, "alpha", jams::defaults::material_alpha)),
            transform(jams::config_optional<jams::Mat<double, 3, 3>>(cfg, "transform", jams::defaults::material_spin_transform)) {

      if (cfg.exists("spin")) {
        const auto& spin_setting = cfg["spin"];
        bool is_sequence = jams::is_sequence_setting(spin_setting);
        bool is_string = jams::is_string_setting(spin_setting);

        if (!(is_sequence || is_string)) {
          throw std::runtime_error("spin setting is not string or array");
        }

        if (is_sequence) {
          auto length = spin_setting.getLength();
          if (length == 3) {
            spin = jams::read_vec_setting<double, 3>(spin_setting, "spin");
          } else if (length == 2) {
            const auto spherical_angles = jams::read_numeric_sequence_setting<double>(spin_setting, "spin", 2);
            spin = jams::spherical_to_cartesian_vector(
                1.0, deg_to_rad(spherical_angles[0]), deg_to_rad(spherical_angles[1]));
          } else {
            throw std::runtime_error("spin setting array is not length 2 or 3");
          }
        } else if (jams::setting_equals_string(spin_setting, "random")) {
            randomize = true;
        }
      }
    }

};
#endif //JAMS_MATERIAL_H
