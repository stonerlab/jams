//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_MATERIAL_H
#define JAMS_MATERIAL_H

#include <string>
#include <libconfig.h++>

#include "jams/helpers/defaults.h"
#include "jams/core/types.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include "jams/helpers/exception.h"

class Material {
public:
    int           id = 0;
    std::string name = "";
    double    moment = 0.0;
    double     gyro  = jams::defaults::material_gyro;
    double     alpha = jams::defaults::material_alpha;
    Vec3       spin  = jams::defaults::material_spin;
    Mat3   transform = jams::defaults::material_spin_transform;
    bool   randomize = false;

    inline Material() = default;

    inline explicit Material(const libconfig::Setting& cfg) :
            id       (0),
            name     (jams::config_required<std::string>(cfg, "name")),
            moment   (jams::config_required<double>(cfg, "moment")),
            gyro     (jams::config_optional<double>(cfg, "gyro", jams::defaults::material_gyro)),
            alpha    (jams::config_optional<double>(cfg, "alpha", jams::defaults::material_alpha)),
            transform(jams::config_optional<Mat3>(cfg, "transform", jams::defaults::material_spin_transform)) {

      if (cfg.exists("spin")) {
        bool is_array = (cfg["spin"].getType() == libconfig::Setting::TypeArray);
        bool is_string = (cfg["spin"].getType() == libconfig::Setting::TypeString);

        if (!(is_array || is_string)) {
          std::runtime_error("spin setting is not string or array");
        }

        if (is_array) {
          auto length = cfg["spin"].getLength();
          if (length == 3) {
            spin = jams::config_required<Vec3>(cfg, "spin");
          } else if (length == 2) {
            spin = spherical_to_cartesian_vector(1.0, deg_to_rad(double(cfg["spin"][0])), deg_to_rad(double(cfg["spin"][1])));
          } else {
            throw std::runtime_error("spin setting array is not length 2 or 3");
          }
        } else if (is_string && lowercase(cfg["spin"]) == "random") {
            randomize = true;
        }
      }
    }

};
#endif //JAMS_MATERIAL_H
