.. _requirements:

Requirements
============

.. note::

    In recent versions of JAMS some of the dependencies are now pulled automagically by cmake and built within the JAMS
    project. This means they do not have to be installed separately. Currently these are: `spglib` and `libconfig`.

- cmake >= 3.8.0
- cuda >= 9.0.176
- blas (can be from Intel MKL)
- fftw3 (can be from Intel MKL)
- hdf5
- libconfig++ >= 1.6.0 (https://hyperrealm.github.io/libconfig/)
- spglib (https://atztogo.github.io/spglib/)
- sphinx (for documentation)
