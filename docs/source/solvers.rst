.. _solvers:

Solvers
=======

Time and duration limits
------------------------

Dynamics solvers read ``t_step``, ``t_min`` and ``t_max`` in seconds and use
picoseconds internally. The clock, timestep, integration-stage times and
Hamiltonian time arguments use double precision, including mixed-precision
builds. Completed-step time is calculated as ``iteration * timestep`` rather
than accumulated by repeated addition.

``t_step`` must be finite and positive; duration limits must be finite and
satisfy ``0 <= t_min <= t_max``. Duration-to-step ratios within
``8 * epsilon(double) * max(1, abs(ratio))`` of an integer are snapped to that
integer. Otherwise ``t_max`` is rounded down and ``t_min`` up to whole steps.
Limits that contain no common timestep boundary, or require a step count
beyond the signed integer range, are rejected. ``t_max = 0`` performs no steps.

Regular monitors sample before each step, including the initial state at zero.
They do not receive an additional sample at the final completed step. Monitor
stride and convergence may further reduce the number of samples; specialised
final-state output remains monitor-specific. Monte Carlo sweep counts and
rotation coordinates retain their existing meanings.


.. toctree::
   :glob:
   :maxdepth: 2

   solvers/*
