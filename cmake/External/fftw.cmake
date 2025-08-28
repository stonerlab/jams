include(CheckCXXSourceCompiles)

add_library(fftw_external INTERFACE)

# First try MKL + its FFTW wrappers
find_package(MKL QUIET)
set(JAMS_FFTW3_FOUND FALSE)

if (MKL_FOUND)
    # Probe whether MKL's FFTW headers actually exist
    find_path(MKL_FFTW_INCLUDE_DIR
            NAMES fftw3.h
            PATHS ${MKL_INCLUDE_DIR}
            PATH_SUFFIXES fftw
    )

    if (MKL_FFTW_INCLUDE_DIR)
        # Try compiling a trivial test using fftw_plan_dft_1d
        set(CMAKE_REQUIRED_INCLUDES ${MKL_FFTW_INCLUDE_DIR})
        set(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES})
        check_cxx_source_compiles("
            #include <fftw3.h>
            int main() {
              fftw_plan p = fftw_plan_dft_1d(4, 0, 0, FFTW_FORWARD, FFTW_ESTIMATE);
              return 0;
            }" MKL_HAS_FFTW)

        if (MKL_HAS_FFTW)
            target_include_directories(fftw_external INTERFACE ${MKL_FFTW_INCLUDE_DIR})
            target_link_libraries(fftw_external INTERFACE ${MKL_LIBRARIES})
            set(JAMS_FFTW3_FOUND TRUE)
            set(JAMS_FFTW3_VENDOR "mkl")
            set(JAMS_FFTW3_LIBRARIES ${MKL_LIBRARIES})
        endif()
    endif()
endif()

# Fallback: look for a real FFTW library
if (NOT JAMS_FFTW3_FOUND)
    find_package(FFTW3 QUIET)
    if (FFTW3_FOUND)
        target_include_directories(fftw_external INTERFACE ${FFTW3_INCLUDE_DIR})
        target_link_libraries(fftw_external INTERFACE ${FFTW3_LIBRARY})
        set(JAMS_FFTW3_FOUND TRUE)
        set(JAMS_FFTW3_VENDOR "fftw3")
        set(JAMS_FFTW3_LIBRARIES ${FFTW3_LIBRARY})
    endif()
endif()