#include <version.h>

#include <spglib.h>
#if HAS_MKL
#include "mkl.h"
#endif

namespace jams {
    namespace build {
        #if HAS_MKL
        std::string mkl_version() {
            MKLVersion Version;
            mkl_get_version(&Version);

            std::stringstream ss;

            ss << Version.MajorVersion << "." << Version.MinorVersion << "." << Version.UpdateVersion;
            ss << " (Build " << Version.Build << ")";

            return ss.str();
        }
        #endif

        std::string spglib_version() {
          return std::to_string(spg_get_major_version()) + "." + std::to_string(spg_get_minor_version()) + "." + std::to_string(spg_get_micro_version());
        }
    }
}