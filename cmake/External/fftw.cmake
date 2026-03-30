include(CheckCXXSourceCompiles)
find_package(Threads QUIET)

add_library(fftw_external INTERFACE)

# First try MKL + its FFTW wrappers
find_package(MKL QUIET)
set(JAMS_FFTW3_FOUND FALSE)
set(JAMS_FFTW3_THREADS_FOUND FALSE)
set(JAMS_FFTW3_THREAD_LIBRARIES)

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

            set(CMAKE_REQUIRED_INCLUDES ${MKL_FFTW_INCLUDE_DIR})
            set(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES})
            check_cxx_source_compiles("
                #include <fftw3.h>
                int main() {
                  if (!fftw_init_threads()) return 1;
                  fftw_plan_with_nthreads(2);
                  fftw_cleanup_threads();
                  return 0;
                }" MKL_HAS_FFTW_THREADS)

            if (MKL_HAS_FFTW_THREADS)
                target_compile_definitions(fftw_external INTERFACE JAMS_HAS_FFTW_THREADS=1)
                set(JAMS_FFTW3_THREADS_FOUND TRUE)
            endif()
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

        find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads PATHS ENV LD_LIBRARY_PATH /usr/lib /usr/local/lib PATH_SUFFIXES)
        find_library(FFTW3_OMP_LIBRARY NAMES fftw3_omp PATHS ENV LD_LIBRARY_PATH /usr/lib /usr/local/lib PATH_SUFFIXES)

        foreach(FFTW3_THREAD_CANDIDATE ${FFTW3_THREADS_LIBRARY} ${FFTW3_OMP_LIBRARY})
            if (FFTW3_THREAD_CANDIDATE AND NOT JAMS_FFTW3_THREADS_FOUND)
                set(CMAKE_REQUIRED_INCLUDES ${FFTW3_INCLUDE_DIR})
                set(CMAKE_REQUIRED_LIBRARIES ${FFTW3_LIBRARY} ${FFTW3_THREAD_CANDIDATE} ${CMAKE_THREAD_LIBS_INIT})
                unset(FFTW3_THREAD_API_WORKS CACHE)
                check_cxx_source_compiles("
                    #include <fftw3.h>
                    int main() {
                      if (!fftw_init_threads()) return 1;
                      fftw_plan_with_nthreads(2);
                      fftw_cleanup_threads();
                      return 0;
                    }" FFTW3_THREAD_API_WORKS)

                if (FFTW3_THREAD_API_WORKS)
                    target_link_libraries(fftw_external INTERFACE ${FFTW3_THREAD_CANDIDATE})
                    if (CMAKE_THREAD_LIBS_INIT)
                        target_link_libraries(fftw_external INTERFACE ${CMAKE_THREAD_LIBS_INIT})
                    endif()
                    target_compile_definitions(fftw_external INTERFACE JAMS_HAS_FFTW_THREADS=1)
                    set(JAMS_FFTW3_THREADS_FOUND TRUE)
                    set(JAMS_FFTW3_THREAD_LIBRARIES ${FFTW3_THREAD_CANDIDATE} ${CMAKE_THREAD_LIBS_INIT})
                endif()
            endif()
        endforeach()
    endif()
endif()
