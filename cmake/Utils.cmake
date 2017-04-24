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
# http://stackoverflow.com/questions/37621342/cmake-will-not-compile-to-c-11-standard
macro(jams_use_cxx11 target)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    if(COMPILER_SUPPORTS_CXX11)
      set (CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
    else()
      message(FATAL_ERROR "${COMPILER_SUPPORTS_CXX11}")
    endif(COMPILER_SUPPORTS_CXX11)
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
