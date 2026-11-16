llg-rs2-gpu
===========

``llg-rs2-gpu`` integrates the Landau–Lifshitz–Gilbert equation on a CUDA
GPU. It combines Ableidinger and Buckwar's Runge–Kutta–Munthe–Kaas
construction with Rößler's RS2 method for Stratonovich equations. Each step
evaluates the Hamiltonian field at three trial configurations and applies a
rotation to each starting spin to preserve its length.
(`Ableidinger and Buckwar <https://doi.org/10.1016/j.apnum.2017.01.017>`_,
`Rößler <https://doi.org/10.1007/s10543-007-0130-3>`_)

.. code-block:: cfg

   solver = {
     module = "llg-rs2-gpu";
     thermostat = "langevin-white-gpu";
     t_step = 1e-15;
     t_max = 1e-12;
   };

The time settings are in seconds. The optional ``gilbert_prefactor`` setting
uses the effective gyromagnetic ratio in the LLG torque when enabled.

Thermostats
-----------

``langevin-white-gpu`` (default)
  The classical bath uses the RS2 white increments and includes the damping
  term in the thermal torque. Use a fixed scalar ``physics.temperature`` or a
  fixed spatial temperature profile configured with
  ``thermostat.temperature_regions``. The noise amplitude uses the bare
  gyromagnetic ratio. The method targets weak order two when white noise is
  active.

``quantum-spde-gpu``
  The SPDE thermostat supplies a persistent coloured field. The initial sample
  comes from the end of any warm-up, the zero-initialised bath, or the first
  stationary sample, according to the thermostat settings.

``general-fft-gpu``
  The FFT thermostat supplies a persistent sampled field for a single material
  at fixed uniform temperature. Its configured spectrum, including a flat
  classical spectrum, enters the drift as a sampled driving field. The solver
  uses standard LLG damping with this prescribed field.

For SPDE and FFT, a single bath sequence advances at intervals of one third of
the solver timestep. The first drift uses the sample at the start of the step.
Both later drifts use the sample two thirds of the way through the step. The
endpoint sample becomes the start sample for the following step. The
deterministic tableau gives third-order pathwise integration for sufficiently
smooth, accurately sampled driving fields.
