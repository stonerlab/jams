Building
========

JAMS is built using the CMake build system. Building `should` be straight
forward, but it depends on several other libraries and packages
(see :ref:`Requirements <requirements>`) which must be built/installed before building JAMS.
These requirements must be installed in a path which is findable for CMake.

JAMS should be built in a separate directory from the source code.
`Debug` or `Release` builds can be built using the usual CMake flags
:code:`-DCMAKE_BUILD_TYPE=Debug` or :code:`-DCMAKE_BUILD_TYPE=Release`.

.. code-block:: none

  mkdir build-release && cd build-release
  cmake -DCMAKE_BUILD_TYPE=Release ..

The :file:`jams` binary will be in the :file:`build-release/bin` directory.

The build process also accepts the following build options:

.. describe:: -DJAMS_BUILD_CUDA=ON

    Toggles CUDA support. Some monitors and solvers are written only in a CUDA version and will be unavailable when
    built with this option.

.. describe:: -DJAMS_BUILD_FASTMATH=ON

    Toggle compiler's fast math optimisations which break strict IEEE 754 compliance. These are mainly concerned with the
    handling of NaNs and Infs and should be safe to enable on most systems.

.. describe:: -DJAMS_BUILD_OMP=OFF

    Toggles OpenMP support in a few functions within JAMS.

.. describe:: -DJAMS_BUILD_MIXED_PREC=OFF

    Toggle some routines to use mixed single/double floating point calculations. Currently none are implemented.

.. describe:: -DJAMS_BUILD_TESTS=OFF

    Toggle building JAMS unit tests.

.. describe:: -DJAMS_BUILD_DOCS=OFF

    Toggle building this documentation.

