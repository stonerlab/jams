# JAMS

Joe's Awesome Magnetic Simulator is an atomistic magnetisation dynamics code with CUDA GPU acceleration.

## Compiling

JAMS is built using the CMake build system. Building `should` be straight forward, but it depends on several other
libraries and packages which also need to be installed or built.

### Dependencies

**Note**: In recent versions of JAMS some of the dependencies are now pulled automagically by cmake and built within the JAMS
project. This means they do not have to be installed separately. Currently these are: `spglib`.

Recent versions of JAMS have the following build requirements:

- cmake >= 3.8
- cuda >= 9.0
- hdf5 >= 1.10 (older versions are API incompatible)
- blas (e.g. mkl, openblas)
- fftw3 (can be supplied by mkl)

Older versions will also need 

- [libconfig++](https://hyperrealm.github.io/libconfig/) >= 1.6.0
- [spglib](http://spglib.github.io/spglib/)

To build documentation we need 

- [sphinx](http://sphinx-doc.org)

### Build Options
CMake accepts the following build options:

##### -DJAMS_BUILD_CUDA=ON

Toggles CUDA support. Some monitors and solvers are written only in a CUDA version and will be unavailable when
    built with this option.

##### -DJAMS_BUILD_FASTMATH=ON

Toggle compiler's fast math optimisations which break strict IEEE 754 compliance. These are mainly concerned with the
    handling of NaNs and Infs and should be safe to enable on most systems.

##### -DJAMS_BUILD_OMP=OFF

Toggles OpenMP support in a few functions within JAMS.

##### -DJAMS_BUILD_MIXED_PREC=OFF

Toggle some routines to use mixed single/double floating point calculations. Currently none are implemented.

##### -DJAMS_BUILD_TESTS=OFF

Toggle building JAMS unit tests.

##### -DJAMS_BUILD_DOCS=OFF

Toggle building this documentation.

### Running CMake

JAMS should be built in a separate directory from the source code. `Debug` or `Release` builds can be built using the
usual CMake flags `-DCMAKE_BUILD_TYPE=Debug` or `-DCMAKE_BUILD_TYPE=Release`.

```shell
	mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release ..
	make -j 10
```

## Running

JAMS runs from an input configuration file

```shell
	./jams input.cfg
```

Settings in the configuration file can also be overwritten or added by include a patch string at the end of the command
line arguments. For example:

```shell
  ./jams input.cfg 'physics : {temperature = 100.0;};'
```

This provides a simple way to write batch scripts to loop over parameters or chain together multiple simulations with
different steps.

## Development

### Integration tests

Use `scripts/run-integration-tests.sh` to create a fresh Python venv, build the JAMS binary, and run the integration tests.

```shell
  bash scripts/run-integration-tests.sh
```

To build a specific variant, pass extra CMake flags:

```shell
  bash scripts/run-integration-tests.sh -o -DJAMS_BUILD_CUDA=OFF -o -DJAMS_BUILD_OMP=OFF
```

Other useful options:

- `--build-type Debug`
- `--generator Ninja`
- `--jobs 8`
- `--enable-gpu`
- `--tests test/test_exchange_symops.py`

### Internal Units

Within the code we use the following units:

- time: picoseconds (ps)
- frequency: terahertz (THz)
- energy: millielectron volts (meV)
- field: Tesla (T)

This gives equations where physical constants are all reasonably close to 1 so 
that the implementation in code can be written exactly the same as the maths on 
paper (i.e. there are no arbitrary normalised units). We avoid Hartree atomic
units because the scales are quite a bit smaller than the relevant spin dynamic 
scales and there's a high potential for bugs, for example because moments are
multiples of $2\mu_B$.
