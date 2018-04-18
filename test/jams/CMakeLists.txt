enable_testing()

include("${PROJECT_SOURCE_DIR}/cmake/External/gtest.cmake")

add_executable(jams_unit_tests "jams_test.cc" $<TARGET_OBJECTS:jams_object_lib>)
target_link_libraries(jams_unit_tests gtest gmock fftw3 cblas jblib pcg symspg config++ hdf5 cusparse curand cublas cufft)
target_compile_features(jams_unit_tests PRIVATE cxx_std_11)
target_compile_definitions(jams_unit_tests PRIVATE HAS_CUDA=1)
target_include_directories(jams_unit_tests PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)
target_include_directories(jams_unit_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_include_directories(jams_unit_tests PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>)

add_test(unit-tests jams_unit_tests)