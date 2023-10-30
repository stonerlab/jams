add_library(cblas_external INTERFACE)

set(BLAS_VENDOR_SEARCH_LIST "MKL" "Apple" "OpenBLAS" "GoTo" "Intel10_64lp_seq" "Intel10_64lp" "Intel10_32" "Intel" "ATLAS" "FLAME" "Generic")

foreach (VENDOR ${BLAS_VENDOR_SEARCH_LIST})
    if("${VENDOR}" STREQUAL "MKL")
        find_package(MKL QUIET)
        if(MKL_FOUND)
            set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
            set(BLAS_LIBRARIES ${MKL_LIBRARIES})
            set(JAMS_CBLAS_FOUND true)
        endif()
    else()
        set(BLA_VENDOR ${VENDOR})
        find_package(BLAS)
        if(BLAS_FOUND)
            set(VENDOR_LOWER $<LOWER_CASE:${VENDOR}>)
            set(VENDOR_UPPER $<UPPER_CASE:${VENDOR}>)
            set(CBLAS_INCLUDE_SEARCH_PATHS
                    $ENV{${VENDOR}_INCLUDE_DIR}
                    $ENV{${VENDOR_LOWER}_INCLUDE_DIR}
                    $ENV{${VENDOR_UPPER}_INCLUDE_DIR}
                    $ENV{${VENDOR}_HOME}
                    $ENV{${VENDOR_LOWER}_HOME}
                    $ENV{${VENDOR_UPPER}_HOME}
                    $ENV{${VENDOR}_ROOT}
                    $ENV{${VENDOR_LOWER}_ROOT}
                    $ENV{${VENDOR_UPPER}_ROOT}
                    $ENV{${VENDOR}_PREFIX}
                    $ENV{${VENDOR_LOWER}_PREFIX}
                    $ENV{${VENDOR_UPPER}_PREFIX}
                    $ENV{BLAS_HOME}
                    $ENV{BLAS_ROOT}
                    /usr/local/include
                    /usr/local/opt
                    /opt
            )

            cmake_path(CONVERT "${CBLAS_INCLUDE_SEARCH_PATHS}" TO_CMAKE_PATH_LIST CBLAS_INCLUDE_SEARCH_PATHS)
            find_path(CBLAS_INCLUDE_DIR NAMES cblas.h HINTS ${CBLAS_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES "${VENDOR}" "${VENDOR_LOWER}" "include" "include/${VENDOR}" "include/${VENDOR_LOWER}" REQUIRED NO_CACHE)
            set(JAMS_CBLAS_FOUND true)
        endif()
    endif()

    if(JAMS_CBLAS_FOUND)
        target_include_directories(cblas_external INTERFACE ${CBLAS_INCLUDE_DIR})
        target_link_libraries(cblas_external INTERFACE ${BLAS_LIBRARIES})
        set(JAMS_CBLAS_VENDOR ${VENDOR})
        break()
    endif()
endforeach ()

if (NOT JAMS_CBLAS_FOUND)
    message(FATAL_ERROR "Failed to find a BLAS library")
endif()
