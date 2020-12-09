add_library(cblas_external INTERFACE)

find_package(MKL QUIET)
if(MKL_FOUND)
    target_include_directories(cblas_external INTERFACE ${MKL_INCLUDE_DIR})
    target_link_libraries(cblas_external INTERFACE ${MKL_LIBRARIES})
    set(JAMS_CBLAS_FOUND true)
    set(JAMS_CBLAS_VENDOR "mkl")
    set(JAMS_CBLAS_LIBRARIES ${MKL_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    target_include_directories(cblas_external INTERFACE ${BLAS_INCLUDE_DIR})
    target_link_libraries(cblas_external INTERFACE ${BLAS_LIBRARIES})
    set(JAMS_CBLAS_FOUND true)
    set(JAMS_CBLAS_VENDOR "other")
    set(JAMS_CBLAS_LIBRARIES ${BLAS_LIBRARIES})
endif(MKL_FOUND)
