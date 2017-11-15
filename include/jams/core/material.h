//
// Created by Joe Barker on 2017/11/15.
//

#ifndef JAMS_MATERIAL_H
#define JAMS_MATERIAL_H

#include <string>
#include <libconfig.h++>

#include "jams/core/defaults.h"
#include "jams/core/types.h"
#include "jams/core/maths.h"
#include "jams/core/utils.h"
#include "jams/core/config.h"

#include "exception.h"

class Material {
public:
    int           id = 0;
    std::string name = "";
    double    moment = 0.0;
    double     gyro  = jams::default_gyro;
    double     alpha = jams::default_alpha;
    Vec3       spin  = jams::default_spin;
    Vec3   transform = {1.0, 1.0, 1.0};
    bool   randomize = false;

    inline Material() = default;

    inline explicit Material(const libconfig::Setting& cfg) :
            id       (0),
            name     (jams::config_required<string>(cfg, "name")),
            moment   (jams::config_required<double>(cfg, "moment")),
            gyro     (jams::config_optional<double>(cfg, "gyro", jams::default_gyro)),
            alpha    (jams::config_optional<double>(cfg, "alpha", jams::default_alpha)),
            transform(jams::config_optional<Vec3>(cfg, "transform", jams::default_spin_transform)) {

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
