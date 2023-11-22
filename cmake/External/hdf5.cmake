add_library(hdf5_external INTERFACE)

set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 COMPONENTS C QUIET)


target_include_directories(hdf5_external INTERFACE ${HDF5_INCLUDE_DIR})
target_link_libraries(hdf5_external INTERFACE ${HDF5_LIBRARIES})
set(JAMS_HDF5_FOUND true)
set(JAMS_HDF5_VERSION ${HDF5_VERSION})
set(JAMS_HDF5_LIBRARIES ${FFTW3_LIBRARY})


set(JAMS_HDF5_VERSION ${HDF5_VERSION})
set(JAMS_HDF5_LIBRARIES ${HDF5_LIBRARIES})

