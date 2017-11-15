function(jams_pickup_jams_sources root)

  # collect files
  file(GLOB_RECURSE srcs ${root}/src/jams/*.cc ${root}/src/jams/*.h)
  file(GLOB_RECURSE cuda ${root}/src/jams/*.cu)

  # convert to absolute paths
  jams_convert_absolute_paths(srcs)
  jams_convert_absolute_paths(cuda)

  # propagate to parent scope
  set(srcs ${srcs} PARENT_SCOPE)
  set(cuda ${cuda} PARENT_SCOPE)
endfunction()