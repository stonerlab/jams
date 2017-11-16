set(VERSION_MAJOR "2" )
set(VERSION_MINOR "0" )
set(VERSION_PATCH "0" )
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S")

configure_file(
        ${CMAKE_SOURCE_DIR}/src/jams/version.h.in
        ${CMAKE_BINARY_DIR}/generated/version.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)

# put source files into 'srcs' and 'cuda'
jams_pickup_jams_sources(${PROJECT_SOURCE_DIR})

#add_definitions("-DVERSION=${VERSION}")
#add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
#add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

add_definitions(-DCUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30") # Kepler K20/K40/K80
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60") # Pascal (P100)


#if(HAVE_CUDA)
	list(APPEND srcs ${cuda})

	cuda_add_executable(jams ${srcs})
	target_link_libraries(jams ${JAMS_LINKER_LIBS})
	jams_use_cxx11(jams)
	#set_property(TARGET jams PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})



#endif()

