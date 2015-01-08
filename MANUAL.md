Title:  JAMS++ User Manual  
Author: Joseph Barker  
Email:  joseph.barker@imr.tohoku.ac.jp 
Date:   5 September 2014

# Introduction


# Configuration File

## sim

`sim.solver` // string // REQUIRED //
: Solver to evolve the simulations. This can be a dynamic integration such as LLG or energy based such as Monte Carlo.
: `HEUNLLG` Heun integration on the CPU
: `CUDAHEUNLLG` -- Heun integration on the GPU
: `METROPOLISMC` -- Metropolis Monte Carlo intengration on the CPU

`sim.t_step` // float (seconds) // REQUIRED //
: Integration time step. For Monte Carlo solvers this should be set to 1.0.

`sim.t_run` // float (seconds) // REQUIRED //
: Total integration run time.

`sim.seed` // int //
: Initial seed for the random number generator. By default this seeds from time. 
: Note: The GPU runs a seperate random number generator. This is seeded from the CPU but means that the random numbers generated for GPU integrators and CPU integrators are not the same set.

---

## lattice

`lattice.size` // [ int, int, int ] // REQUIRED //
: Number of unit cells along each basis vector.

`lattice.periodic` // [ bool, bool, bool ] // REQUIRED //
: Enable periodic boundaries along each basis vector.

`lattice.parameter` // float (nm) // REQUIRED //
: Lattice parameter.

`lattice.basis` // ( [ float, float, float ] x3 ) // REQUIRED //
: Lattice basis vectors (see below)


	basis = (
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]);


`lattice.positions` // string (file path) // REQUIRED //
: File path of the atomic positions file.

`lattice.exchange` // string (file path) // REQUIRED //
: File path of the exchange interactions file.

`lattice.spins` // string (file path) // 
: File path of initial spin configuration from a JAMS .h5 file.

---

## materials

`materials.[n].name` // string // REQUIRED //
: Unique name for material to be referenced in the `lattice.positions` input file.

`materials.[n].moment` // float (Bohr magnetons) // REQUIRED //
: Magnetic moment

`materials.[n].alpha` // float // REQUIRED //
: Gilbert damping parameter

`materials.[n].gyro` // float (𝛾~e~) // default 1.0 //
: Gyromagnetic ratio

`materials.[n].spin` // string OR [float, float] OR [float, float, float] // default [ 0.0, 0.0, 1.0 ] //
: Initial spin value for this material. This setting is overridden if an initial spin file is given in the `lattice` section.
: Options: 
: "RANDOM" -- Randomize each spin of this material type
: [ S~θ~, S~φ~ ] -- Spherical spin components in degrees, θ is polar angle, φ is azimuthal angle.
: [ S~x~, S~y~, S~z~ ] -- Cartesian spin components

---

## solvers

#### constrainedmc

`cmc_constraint_theta` // float (deg) // REQUIRED //
: Polar magnetization constraint angle θ

`cmc_constraint_phi` // float (deg) // REQUIRED //
: Azimuthal magnetization constraint angle φ


`sigma` // float // default 0.05 //
: Trial move size (width of Gaussian)

### monitors

#### hdf5

This module outputs binary data in hdf5 format files (.h5).

##### hdf5.float_type (string)

The output precision of floating point data. The options are `float` or `double` for 32 bit and 64 bit floating point formats respectively. The data is stored as IEEE little endian binary.

##### hdf5.compressed (bool)

Toggle zlib compression of the h5 file. 

Because the data is stored in binary format this does not usually give a large reduction is size. If high accuracy is not needed then storing in *float* instead of *double* will roughly half the size.


# JAMS++ Developer Manual

## Design overview

JAMS++ is designed in a modular fashion. The primary module is a _solver_ module. Solvers may be dynamic (e.g. Langevin LLG) or static (e.g. Metropolis Monte-Carlo) but they all evolve the spin system in some respect. The physical conditions (temperature, applied field etc.) are specified by a _physics_ module. A single physics module is registered to the solver class which will call a physics update every 'iteration'. The concept of 'observer' _monitor_ modules is also featured. Multiple monitors can be registered with the solver module which will update them every iteration. Monitors usually generate output, although this will usually not be output every iteration, although the statistics may require frequent updating.

## Solver modules

## Physics modules

### Template

    // Copyright 2014 Joseph Barker. All rights reserved.

    #ifndef JAMS_PHYSICS_MYPACKAGE_H
    #define JAMS_PHYSICS_MYPACKAGE_H

    #include <libconfig.h++>

    #include "core/physics.h"

    class MyPackagePhysics : public Physics {

     public:
      // constructor (must call base constructor)
      MyPackagePhysics(const libconfig::Setting &settings)
       : Physics(settings) {}

      ~MyPackagePhysics() {};

      // update is called by the solver every iteration
      void update(const int &iterations, const double &time, const double &dt) {};

     private:
       // private members
    };

    #endif  // JAMS_PHYSICS_MYPACKAGE_H

### Generating output

The base class has a member `output_step_freq_` which is set from the config `physics.output_steps`. This can be used in the update routine to check if output should be generated from the physics package (independantly of any monitors).

## Monitor modules