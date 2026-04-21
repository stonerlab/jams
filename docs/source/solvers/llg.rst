LLG Solvers
===========

The LLG solver family integrates the Landau-Lifshitz-Gilbert equation

.. math::

   \frac{d\vec{S}_i}{dt} =
   -\gamma_i \left(
     \vec{S}_i \times \vec{H}_i
     + \alpha_i \vec{S}_i \times (\vec{S}_i \times \vec{H}_i)
     + \frac{1}{\mu_i}\vec{S}_i \times (\vec{S}_i \times \vec{\tau}_i)
   \right),

where :math:`\vec{\tau}_i` is the sum of any extra deterministic torque terms
configured in ``dynamics.terms``.

Backends and integrators
------------------------

Set ``dynamics.equation = "llg"`` and choose one of the following
integrators:

.. list-table:: LLG integrators
   :header-rows: 1

   * - Integrator
     - CPU alias
     - GPU alias
     - Notes
   * - ``heun``
     - ``llg-heun-cpu``
     - ``llg-heun-gpu``
     - Predictor-corrector scheme
   * - ``rk4``
     - ``llg-rk4-cpu``
     - ``llg-rk4-gpu``
     - Classical fourth-order Runge-Kutta
   * - ``rkmk2``
     - ``llg-rkmk2-cpu``
     - ``llg-rkmk2-gpu``
     - Second-order Runge-Kutta-Munthe-Kaas
   * - ``rkmk4``
     - ``llg-rkmk4-cpu``
     - ``llg-rkmk4-gpu``
     - Fourth-order Runge-Kutta-Munthe-Kaas
   * - ``simp``
     - ``llg-simp-cpu``
     - ``llg-simp-gpu``
     - Semi-implicit Cayley update
   * - ``dm``
     - ``llg-dm-cpu``
     - ``llg-dm-gpu``
     - Depondt-Mertens scheme

Example
-------

.. code-block:: cfg

   solver = {
     backend = "gpu";
     integrator = "rkmk4";
     t_step = 1.0e-15;
     t_max = 1.0e-9;
   };

   dynamics = {
     equation = "llg";
   };

Required settings
-----------------

``solver.t_step``
  Integration time step in seconds.

``solver.t_max``
  Maximum integration time in seconds.

Optional settings
-----------------

``solver.t_min``
  Minimum integration time in seconds. The solver will continue to this time
  even if a monitor converges earlier.

``solver.gilbert_prefactor``
  Toggle the Gilbert-prefactor form in the stochastic noise amplitude.
  Default: ``false``.

Temperature and thermostats
---------------------------

At ``physics.temperature = 0`` the dynamics are deterministic.

For ``physics.temperature > 0``:

- CPU LLG solvers use the built-in classical white-noise thermal field.
- GPU LLG solvers use the configured CUDA thermostat. The default is
  ``solver.thermostat = "langevin-white-gpu"``.

Additional deterministic torques
--------------------------------

Spin-transfer and spin-orbit torques are configured through
``dynamics.terms``. See :doc:`configuration` for the full syntax.

Example: SOT on all spins
-------------------------

.. code-block:: cfg

   solver = {
     backend = "cpu";
     integrator = "rk4";
     t_step = 1.0e-15;
     t_max = 1.0e-9;
   };

   dynamics = {
     equation = "llg";
     terms = (
       {
         module = "sot";
         spin_polarisation = [0.0, 1.0, 0.0];
         spin_hall_angle = 0.1;
         charge_current_density = 1.0e12;
       }
     );
   };

Example: STT only on surface spins
----------------------------------

.. code-block:: cfg

   dynamics = {
     equation = "llg";
     terms = (
       {
         module = "stt";
         coefficient = 2.0;
         spin_polarisation = [0.0, 0.0, 1.0];
         selector = {
           surface_layers = 1;
         };
       }
     );
   };

Legacy SOT aliases
------------------

The legacy modules

- ``llg-sot-rk4-cpu``
- ``llg-sot-rk4-gpu``

are still accepted for backward compatibility. They behave like
``integrator = "rk4"`` with ``equation = "llg"`` plus a single legacy SOT
term configured from ``spin_polarisation``, ``spin_hall_angle`` and
``charge_current_density`` in the ``solver`` block.
