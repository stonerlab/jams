Title:  JAMS++ User Manual  
Author: Joseph Barker  
Email:  joseph.barker@imr.tohoku.ac.jp 
Date:   2015-08-18

# Introduction


# Configuration File

Options in **`BOLD`** are up-to-date and are supported, options `NOTBOLD` are NOT up-to-date and may or may not work!

## sim

`sim.solver` // string // REQUIRED //
: Solver to evolve the simulations. This can be a dynamic integration such as LLG or energy based such as Monte Carlo.
: **`CUDAHEUNLLG`** -- Heun integration on the GPU
: `HEUNLLG` -- Heun integration on the CPU
: `METROPOLISMC` -- Metropolis Monte Carlo integration on the CPU

`sim.t_step` // float (seconds) // REQUIRED //
: Integration time step. For Monte Carlo solvers this should be set to 1.0.

`sim.t_run` // float (seconds) // REQUIRED //
: Total integration run time.

`sim.t_run` // float (seconds) // REQUIRED //
: DEPRECIATED -- equilibration time before monitors are used.

`sim.seed` // int //
: Initial seed for the random number generator. By default this seeds from time. 
: Note: The GPU runs a separate random number generator. This is seeded from the CPU but means that the random numbers generated for GPU integrators and CPU integrators are not the same set.

`sim.verbose`  // bool //
: Output extra information to the command line or output file. Mostly used for debugging or finding mistakes in input files.

---

## physics

Physics modules allow specialized complex simulations to be performed. In the minimal form the module defines the temperature and applied magnetic field. Specific modules can generalize this, for example to include a time dependence.

`physics.module` // string // REQUIRED //
: Physics module to use for the simulation
: **`EMPTY`** -- no special physics
: `FMR` -- ferromagnetic resonance -- AC field
: `MFPT` -- mean first passage time
: `TTM` -- two temperature model -- for simulating laser heating
: `SQUARE` -- square field pulses
: `FIELDCOOL` -- field cooling -- change field and temperature as a linear function of time

#### EMPTY

`physics.temperature` // float (Kelvin) // REQUIRED //
: Thermostat temperature

`physics.applied_field` // [ float, float, float ]
: External applied magnetic field as $H_x$, $H_y$, $H_z$

---

### monitors

`monitors.[n].module` // string // REQUIRED //
: **`magnetisation`** -- 
: **`structurefactor`**
: **`hdf5`**

`monitors.[n].output_steps` // int // REQUIRED //
: Number of timesteps between monitor output.
: This can have a large effect on run time because the spin configuration usually has to be copied from the GPU every time a monitor is run for output.

#### magnetisation

This module calculates the reduced magnetisation of each material. The output is written to *config*_mag.dat with the following format where time is in seconds, temperature is Kelvin, Hx,Hy,Hz are applied field in Tesla and mx,my,mz,|m| are the reduced magnetisation values (from -1 to 1). The magnetisation columns are given for each material in order.

    time temperature Hx Hy Hz mx my mz |m| ....

#### structurefactor

This module calculates the dynamical structure factor which is the space time Fourier transform of the spin system.

`monitors.[n].brillouin_zone` // ( [int, int, int], [int, int, int], ...) // REQUIRED
: The path in k-space along which to output the structure factor. Currently these are in Cartesian vectors (i.e. must be multiplied by the inverse lattice vectors to give the path k-space). This will be fixed in a future version.

#### hdf5

This module outputs binary data in hdf5 format files (.h5).

`monitors.[n].float_type` // string
: The output precision of floating point data. The options are `float` or `double` for 32 bit and 64 bit floating point formats respectively. The data is stored as IEEE little endian binary.

`monitors.[n].compressed` // bool
: Toggle zlib compression of the h5 file. 

Because the data is stored in binary format this does not usually give a large reduction is size. If high accuracy is not needed then storing in *float* instead of *double* will roughly half the size.

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


`lattice.atom_positions` // string (file path) // REQUIRED //
: File path of the atomic positions file. File format is as follows where material is the string name and rx, ry, rz are the fractional coordinates in the lattice vectors.


    Material  rx   ry  rz


`lattice.spins` // string (file path) // 
: File path of initial spin configuration from a JAMS .h5 file.

---

## materials

`materials.[n].name` // string // REQUIRED //
: Unique name for material to be referenced in the `lattice.atom_positions` input file.

`materials.[n].moment` // float (Bohr magnetons) // REQUIRED //
: Magnetic moment

`materials.[n].alpha` // float // REQUIRED //
: Gilbert damping parameter

`materials.[n].gyro` // float ($\gamma_e$) // default 1.0 //
: Gyromagnetic ratio

`materials.[n].spin` // string OR [float, float] OR [float, float, float] // default [ 0.0, 0.0, 1.0 ] //
: Initial spin value for this material. This setting is overridden if an initial spin file is given in the `lattice` section.
: Options: 
: "RANDOM" -- Randomize each spin of this material type
: [ $S_{\theta}$, $S_{\phi}$ ] -- Spherical spin components in degrees, $\theta$ is polar angle, $\phi$ is azimuthal angle.
: [ $S_x$, $S_y$, $S_z$ ] -- Cartesian spin components

`materials.[n].transform` // [float, float, float] //
: Specify a transformation to apply to the spin during the structure factor calculation. This can be used for example to flip an opposite orientated antiferromagnetic lattice.


---

## solvers

#### constrainedmc

`cmc_constraint_theta` // float (deg) // REQUIRED //
: Polar magnetization constraint angle $\theta$

`cmc_constraint_phi` // float (deg) // REQUIRED //
: Azimuthal magnetization constraint angle $\phi$


`sigma` // float // default 0.05 //
: Trial move size (width of Gaussian)

---

## hamiltonians

`hamiltonians.[n].module` // string // REQUIRED //
: Name of Hamiltonian module to use
: **`uniaxial`** -- uniaxial anisotropy along z-direction
: **`exchange`** -- exchange interactions

#### uniaxial

The anisotropy can only be specified in terms of K's or d's, not both. The values are specified in the list as one per material.

`K1` // [ float, ...] //
: micromagnetic aniostropy constant $K_1 \sin^{2}\theta$, positive is easy axis
: **Sign convention -- positive is uniaxial**

`K2` // [ float, ...] //
: micromagnetic aniostropy constant $K_2 \sin^{4}\theta$, positive is easy axis
: **Sign convention -- positive is uniaxial**

`K3` // [ float, ...] //
: micromagnetic aniostropy constant $K_3 \sin^{6}\theta$, positive is easy axis
: **Sign convention -- positive is uniaxial**

`d2` // [ float, ...] //
: magnetocyrstalline anisotropy constant $\kappa_{2} (3\cos^{2}\theta - 1) / 2$
: **Sign convention -- negative is uniaxial**

`d4` // [ float, ...] //
: magnetocyrstalline anisotropy constant $\kappa_{4} (35\cos^{4}\theta - 30 \cos^{2}\theta + 3) / 8$

`d6` // [ float, ...] //
: magnetocyrstalline anisotropy constant $\kappa_{6} (231\cos^{6}\theta - 315\cos^{4}\theta + 105\cos^{2}\theta - 5) / 16$

#### exchange

`exc_file` // string (file path) // REQUIRED //
: Path for exchange interactions. The file has the following format where rx,ry,rz are in Cartesian coordinates (one per symmetric interaction) and Jij is in Joules and the materials are the string names. 
: **Sign convention -- positive is ferromagnetic**


    MaterialA   MaterialB  rx   ry  rz  Jij


`exc_file` // string (file path) // 
: Alternatively the full exchange tensor can be specified for each interaction which allows anisotropy or Dzyaloshinskii-Moriya interaction to be included.


    MaterialA   MaterialB  rx   ry  rz  Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz


---

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