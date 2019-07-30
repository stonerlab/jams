Compiling
=========

JAMS is built using the CMake build system. Building `should` be straight forward, but it depends on several other
libraries and packages which also need to be installed or built.

Dependencies
############

.. note::

    In recent versions of JAMS some of the dependencies are now pulled automagically by cmake and built within the JAMS
    project. This means they do not have to be installed separately. Currently these are: `spglib`.

- cmake >= 3.8.0
- cuda >= 9.0.176
- libconfig++ >= 1.6.0
- hdf5
- blas
- spglib
- fftw3
- sphinx (for documentation)

Build Options
#############

CMake accepts the following build options:

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


Running CMake
#############

JAMS should be built in a separate directory from the source code. `Debug` or `Release` builds can be built using the
usual CMake flags :code:`-DCMAKE_BUILD_TYPE=Debug` or :code:`-DCMAKE_BUILD_TYPE=Release`.

.. code-block:: none

  mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..