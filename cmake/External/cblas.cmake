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

            # The FindBLAS package doesn't look for the cblas.h file we need.
            # We create a set of search paths to look for this. First we'll
            # base this on the link libraries which have been found. These
            # are often of the form "/opt/<package_name>/lib/<library>.so"
            # so we will add paths one and two levels up, i.e.:
            # - /opt/<package_name>/lib
            # - /opt/<package_name>
            # We will then also add paths which are specified in environment
            # variables. This is commonly done on HPC clusters with module
            # environments. We're combining this with the vendor names that
            # cmake provides and include normal case, lower case and upper case
            # possibilities. Finally, the `find_path` command will search the
            # default paths in cmake.
            # Additionally we set PATH_SUFFIXES to allow lots of combinations
            # of vendor names and "include".

            list(GET BLAS_LIBRARIES 0 BLAS_LIBRARIES_FIRST_PATH)
            # going up one parent usually leaves us in a 'lib' directory
            cmake_path(GET BLAS_LIBRARIES_FIRST_PATH PARENT_PATH BLAS_LIBRARIES_FIRST_PATH_PARENT)
            # going up two paths hopefully puts us in the base path for the package
            cmake_path(GET BLAS_LIBRARIES_FIRST_PATH_PARENT PARENT_PATH BLAS_LIBRARIES_FIRST_PATH_PARENT2)

            string(TOLOWER "${VENDOR}" VENDOR_LOWER)
            string(TOUPPER "${VENDOR}" VENDOR_UPPER)

            set(CBLAS_INCLUDE_SEARCH_PATHS
                    ${BLAS_LIBRARIES_FIRST_PATH_PARENT}
                    ${BLAS_LIBRARIES_FIRST_PATH_PARENT2}
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

            set(CBLAS_INCLUDE_PATH_SUFFIXES "${VENDOR}" "${VENDOR_LOWER}" "include" "include/${VENDOR}" "include/${VENDOR_LOWER}")
            cmake_path(CONVERT "${CBLAS_INCLUDE_PATH_SUFFIXES}" TO_CMAKE_PATH_LIST CBLAS_INCLUDE_PATH_SUFFIXES)

            find_path(CBLAS_INCLUDE_DIR NAMES cblas.h HINTS ${CBLAS_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES ${CBLAS_INCLUDE_PATH_SUFFIXES} REQUIRED NO_CACHE)
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
