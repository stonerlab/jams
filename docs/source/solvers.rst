.. _solvers:
Solvers
=======

JAMS supports two solver configuration styles:

1. The preferred split configuration using ``solver.backend``,
   ``solver.integrator`` and ``dynamics.equation``.
2. The legacy ``solver.module`` names such as ``llg-rkmk4-gpu``.

Both forms are currently accepted. The split form is the recommended one for
new inputs because it cleanly separates the numerical integrator from the
equation of motion and any extra deterministic spin-torque terms.

.. toctree::
   :maxdepth: 1

   solvers/configuration
   solvers/llg
   solvers/gse-rk4
   solvers/ll-lorentzian-rk4
   solvers/null
   solvers/rotations-cpu
   solvers/monte-carlo-metropolis-cpu
   solvers/monte-carlo-constrained-cpu
   solvers/monte-carlo-metadynamics-cpu
