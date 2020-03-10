//
// Created by Joseph Barker on 2020-03-10.
//

#ifndef JAMS_HELPERS_FILESYSTEM_H
#define JAMS_HELPERS_FILESYSTEM_H

#include <string>
#include <sys/stat.h>
#include "jams/helpers/defaults.h"

namespace jams {
    void make_path(const std::string& path, mode_t mode = jams::defaults::make_path_mode);
}


#endif //JAMS_HELPERS_FILESYSTEM_H
