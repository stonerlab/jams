//
// Created by Joseph Barker on 2020-03-11.
//

#ifndef JAMS_INTERFACE_SYSTEM_H
#define JAMS_INTERFACE_SYSTEM_H

#include <string>
#include "jams/helpers/defaults.h"

namespace jams {
    namespace system {
        void make_directory(const std::string &path, mode_t mode);
        void make_path(const std::string &path, mode_t mode = defaults::make_path_mode);

        bool stdout_is_tty();
    }
}

#endif //JAMS_INTERFACE_SYSTEM_H
