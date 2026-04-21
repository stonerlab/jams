LL Lorentzian RK4 Solvers
=========================

The LL Lorentzian solver integrates a Landau-Lifshitz equation coupled to a
Lorentzian memory process using RK4. In split configuration form use

.. code-block:: cfg

   solver = {
     backend = "gpu";
     integrator = "rk4";
     t_step = 1.0e-15;
     t_max = 1.0e-9;
   };

   dynamics = {
     equation = "ll-lorentzian";
   };

Legacy aliases are:

- ``ll-lorentzian-rk4-cpu``
- ``ll-lorentzian-rk4-gpu``

Required settings
-----------------

``solver.t_step``
  Integration time step in seconds.

``solver.t_max``
  Maximum integration time in seconds.

``thermostat.lorentzian_gamma``
  Lorentzian damping parameter in the thermostat block.

``thermostat.lorentzian_omega0``
  Lorentzian characteristic frequency in the thermostat block.

Optional settings
-----------------

``solver.t_min``
  Minimum integration time in seconds.

GPU usage
---------

The GPU implementation is typically used together with
``solver.thermostat = "langevin-lorentzian-gpu"`` or another compatible
CUDA general-FFT thermostat configuration.

CPU usage
---------

The CPU implementation exposes the same solver/equation interface and reads
the same Lorentzian parameters from the ``thermostat`` block. Unlike the GPU
implementation, CPU stochastic updates currently use the built-in white-noise
thermal field when ``physics.temperature > 0``.
