# Ensures that we do an out of source build

macro(jams_ensure_out_of_source_build)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${CMAKE_BINARY_DIR}" insource)
     get_filename_component(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
     string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${PARENTDIR}" insourcesubdir)
    if(insource OR insourcesubdir)
        message(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build.")
    endif(insource OR insourcesubdir)
endmacro()

################################################################################################
# http://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake
macro(jams_use_cxx11 arg)
  if (CMAKE_VERSION VERSION_LESS "3.1")
  	target_compile_features(${arg} PRIVATE cxx_range_for)
  else ()
  	set_property(TARGET ${arg} PROPERTY CXX_STANDARD 11)
  endif ()
endmacro()

################################################################################################

macro (jams_add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SRCS "${_relPath}/${_src}")
        else()
            list (APPEND SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (SRCS ${SRCS} PARENT_SCOPE)
    endif()
endmacro()

################################################################################################
# Converts all paths in list to absolute
# Usage:
#   jams_convert_absolute_paths(<list_variable>)
function(jams_convert_absolute_paths variable)
  set(__dlist "")
  foreach(__s ${${variable}})
    get_filename_component(__abspath ${__s} ABSOLUTE)
    list(APPEND __list ${__abspath})
  endforeach()
  set(${variable} ${__list} PARENT_SCOPE)
endfunction()