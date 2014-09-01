# JAMS++ User Manual

## Configuration

### sim

This section contains simulation related options.

    sim : {
        // solvers and output settings
    };

    physics : {
        module = "physics_name";
        // options to pass to the physics package
        temperature = 20.0;
        applied_field = [0.0, 0.0, 0.0];
    };

    monitors = (
        {
            module = "monitor_name";
            // number of timesteps between output
            output_steps = 10;
        }
    );

    lattice : {
        // definition of the system lattice
    };

    materials = (
        // definition of material properties
    );

---

##### sim.solver - required

The solver which will be used for the spin system

`HEUNLLG` - default two step Heun integration on the CPU

`CUDAHEUNLLG` - default two step Heun integration on the GPU

`METROPOLISMC` - Metropolis Monte Carlo integration on the CPU

##### sim.seed
Initial seed for the random number generator.

##### sim.save_state

Toggle to save the final spin state in a binary file (bool).

##### sim.t_step (_required_)

Integration timestep (seconds).

##### sim.t_eq (_required_)

Time to equilbrate the system (seconds).

##### sim.t_run (_required_)

Time to run the system monitors (seconds).
---

### lattice

This section contains lattice related options.

---

##### lattice.size (_required_)

Number of unit cells to produce in each basis vector ([int, int, int]).


##### lattice.kpoints (_required_)

Number kpoints in the unitcell for each (reprocal) basis vector ([int, int, int]).

##### lattice.periodic (_required_)

Toggle periodic boundaries along each basis vector ([bool, bool, bool]).

##### lattice.parameter (_required_)

Lattice parameter (nanometers).

##### lattice.basis (_required_)

Lattice basis vectors (see below)

    basis = (
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]);


##### lattice.positions (_required_)

Lattice positions file ("file path").

##### lattice.exchange (_required_)

Exchange interactions file ("file path").

---

### materials

This section contains material related options.

---

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