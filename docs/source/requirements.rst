.. _requirements:

Requirements
============

In recent versions of JAMS some of the dependencies are now pulled automagically by cmake and built within the JAMS
project. This means they do not have to be installed separately.

Automatically built in dependencies:

- libconfig (https://github.com/hyperrealm/libconfig)
- spglib (https://github.com/spglib/spglib)
- highfive (https://github.com/highfive-devs/highfive)
- pcg (https://github.com/imneme/pcg-cpp)

Some standard external libraries are required and should be installed a configured on your system before installing JAMS.
On HPC systems these will often be available through the modules system (https://modules.readthedocs.io/en/stable/modulefile.html)
or you may be able to install in userspace using spack (https://spack.readthedocs.io/en/latest/).

External requirements:

- c++17
- cmake >= 3.10.0
- cuda >= 9.0.176
- blas (can be from Intel MKL)
- fftw3 (can be from Intel MKL)
- hdf5 >= 1.10