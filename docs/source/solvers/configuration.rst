Solver Configuration
====================

Preferred split configuration
-----------------------------

The preferred way to configure a solver is to specify the backend and
integrator in the ``solver`` block, and the equation in the ``dynamics``
block:

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

The available backends are:

- ``cpu``
- ``gpu``

The available integrators are:

- ``null``
- ``rotations``
- ``heun``
- ``rk4``
- ``rkmk2``
- ``rkmk4``
- ``simp`` or ``semi-implicit``
- ``dm``
- ``monte-carlo-metropolis``
- ``monte-carlo-constrained``
- ``monte-carlo-metadynamics``

The currently supported equations are:

- ``llg``
- ``gse``
- ``ll-lorentzian``

Legacy ``solver.module`` aliases
--------------------------------

Older inputs can still use legacy combined names such as ``llg-rk4-gpu``.
These are translated internally to the split configuration.

.. list-table:: Legacy aliases
   :header-rows: 1

   * - Legacy module
     - Split configuration
   * - ``null``
     - ``backend = "cpu"``, ``integrator = "null"``
   * - ``rotations-cpu``
     - ``backend = "cpu"``, ``integrator = "rotations"``
   * - ``llg-heun-cpu`` / ``llg-heun-gpu``
     - ``integrator = "heun"``, ``equation = "llg"``
   * - ``llg-rk4-cpu`` / ``llg-rk4-gpu``
     - ``integrator = "rk4"``, ``equation = "llg"``
   * - ``llg-rkmk2-cpu`` / ``llg-rkmk2-gpu``
     - ``integrator = "rkmk2"``, ``equation = "llg"``
   * - ``llg-rkmk4-cpu`` / ``llg-rkmk4-gpu``
     - ``integrator = "rkmk4"``, ``equation = "llg"``
   * - ``llg-simp-cpu`` / ``llg-simp-gpu``
     - ``integrator = "simp"``, ``equation = "llg"``
   * - ``llg-dm-cpu`` / ``llg-dm-gpu``
     - ``integrator = "dm"``, ``equation = "llg"``
   * - ``gse-rk4-cpu`` / ``gse-rk4-gpu``
     - ``integrator = "rk4"``, ``equation = "gse"``
   * - ``ll-lorentzian-rk4-cpu`` / ``ll-lorentzian-rk4-gpu``
     - ``integrator = "rk4"``, ``equation = "ll-lorentzian"``
   * - ``llg-sot-rk4-cpu`` / ``llg-sot-rk4-gpu``
     - ``integrator = "rk4"``, ``equation = "llg"``, plus a legacy SOT term
   * - ``monte-carlo-metropolis-cpu``
     - ``backend = "cpu"``, ``integrator = "monte-carlo-metropolis"``
   * - ``monte-carlo-constrained-cpu``
     - ``backend = "cpu"``, ``integrator = "monte-carlo-constrained"``
   * - ``monte-carlo-metadynamics-cpu``
     - ``backend = "cpu"``, ``integrator = "monte-carlo-metadynamics"``

LLG dynamics terms
------------------

For the LLG family, extra deterministic torques are configured in the
``dynamics.terms`` list rather than by creating separate solver classes.
Each term contributes an additive per-spin torque which is then integrated by
the selected LLG scheme.

Supported term modules are:

- ``stt``
- ``slonczewski``
- ``sot``

``stt`` and ``slonczewski`` require:

- ``coefficient``
- ``spin_polarisation``

``sot`` requires:

- ``spin_polarisation``
- ``spin_hall_angle``
- ``charge_current_density``

Example:

.. code-block:: cfg

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

Selectors
---------

Every LLG dynamics term can optionally be restricted to a subset of spins by
adding a ``selector`` block.

Supported selector keys are:

- ``material``
- ``materials``
- ``basis_site``
- ``basis_sites``
- ``sites``
- ``surface_layers``

These keys are combined as logical ``AND`` conditions. For example:

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

This evolves surface spins with ``LLG + STT`` while the remaining spins are
evolved with plain ``LLG``.
