add_library(hdf5_external INTERFACE)

find_package(HDF5 COMPONENTS CXX QUIET REQUIRED)
target_include_directories(hdf5_external INTERFACE ${HDF5_INCLUDE_DIRS})
target_link_libraries(hdf5_external INTERFACE ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

set(JAMS_HDF5_VERSION ${HDF5_VERSION})
set(JAMS_HDF5_LIBRARIES ${HDF5_LIBRARIES})

