Building
========

JAMS is built using the CMake build system. Building `should` be straight
forward, but it depends on several other libraries and packages
(see :ref:`Requirements <requirements>`) which must be built/installed before building JAMS.
These requirements must be installed in a path which is findable for CMake.


Build an executable
-------------------
Once the requirements are installed, the simplest way to install JAMS is to download the build
script (https://github.com/stonerlab/jams/blob/master/scripts/build-jams.sh) and run it. You should
have any dependencies visible in your environment (for example by loading modules). You can download and
run the build script in one line (depends on curl):

.. code-block:: bash

  curl -L -o build-jams.sh https://raw.githubusercontent.com/stonerlab/jams/refs/heads/master/scripts/build-jams.sh && bash ./build-jams.sh

The script will automatically look for :code:`nvcc` and build with CUDA support if present.

The script can be invoked with different options (see :code:`bash build-jams.sh -h`) most useful of which is the ability to build
different branches, such as :code:`bash build-jams.sh -b metadynamics` would build a branch called `metadynamics` from the
github repository.

Building from source
--------------------
For development work you will need to fork/clone/download the JAMS source code and build.

JAMS should be built in a separate directory from the source code.
`Debug` or `Release` builds can be built using the usual CMake flags
:code:`-DCMAKE_BUILD_TYPE=Debug` or :code:`-DCMAKE_BUILD_TYPE=Release`.

.. code-block:: none

  mkdir build-release && cd build-release
  cmake -DCMAKE_BUILD_TYPE=Release ..

.. note::

If your CMake is version 4, you may also need to add the backwards compatibility option :code:`-DCMAKE_POLICY_VERSION_MINIMUM=3.10`.

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

    Toggle some routines to use mixed single/double floating point calculations. This is a very old option from the early
    CUDA days. Currently there is no mixed precision in JAMS.

.. describe:: -DJAMS_BUILD_TESTS=OFF

    Toggle building JAMS unit tests.

.. describe:: -DJAMS_BUILD_DOCS=OFF

    Toggle building this documentation.

