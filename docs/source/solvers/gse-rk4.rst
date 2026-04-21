GSE RK4 Solvers
===============

The GSE solver integrates the Gilbert stochastic equation using an RK4 scheme.
In split configuration form use

.. code-block:: cfg

   solver = {
     backend = "cpu";
     integrator = "rk4";
     t_step = 1.0e-15;
     t_max = 1.0e-9;
   };

   dynamics = {
     equation = "gse";
   };

Legacy aliases are:

- ``gse-rk4-cpu``
- ``gse-rk4-gpu``

Required settings
-----------------

``solver.t_step``
  Integration time step in seconds.

``solver.t_max``
  Maximum integration time in seconds.

Optional settings
-----------------

``solver.t_min``
  Minimum integration time in seconds.

``solver.gilbert_prefactor``
  Toggle the Gilbert-prefactor form in the stochastic noise amplitude.

Temperature handling
--------------------

- On CPU, thermal noise is provided by the built-in classical white-noise
  field when ``physics.temperature > 0``.
- On GPU, the stochastic field is provided by the configured CUDA thermostat.
