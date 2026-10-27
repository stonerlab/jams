Spin rate
=========

The ``spin-rate`` monitor reports ``max_ds_dt`` in ps\ :sup:`-1`: the maximum
magnitude of the deterministic local LLG derivative over all magnetic sites.
Vacancies are excluded. Opposing derivatives on different sites do not cancel.

Supported solvers are ``llg-heun-cpu``, ``llg-heun-gpu``, ``llg-rkmk2-cpu``,
``llg-rkmk2-gpu``, ``llg-rkmk4-cpu``, ``llg-rkmk4-gpu`` and ``llg-rk4-gpu``.
Other solvers, including specialised RK4 variants, reject this monitor during
initialisation. An empty magnetic selection or non-finite rate is an error.

.. code-block:: none

   monitors = (
     {
       module = "spin-rate";
       output_steps = 100;
       convergence = 1.0e-5;  // ps^-1; choose for the physical problem
       consecutive_samples = 5;
     }
   );

``convergence`` is an optional finite, positive threshold. Without it, the
monitor only writes diagnostic output. ``consecutive_samples`` is a positive
integer (default 1). A measurement must be strictly below the threshold;
equality or an excursion above it resets the count. Only fresh measurements
count, and satisfaction is not permanently latched.

``output_steps`` sets both checking and output cadence (default 100, positive).
``start_delay_steps`` delays the first eligible output (default 0, non-negative);
sampling still occurs at multiples of ``output_steps``. ``precision`` controls
rate output precision (default 8). Output uses the usual monitor TSV naming and
solver time coordinates in picoseconds.

Measurements before ``solver.t_min`` are written but do not count towards
convergence. Confirming samples start at the first sampling boundary at or
after that minimum. With ``t_min = 0`` and ``consecutive_samples = 1``, an
initially stationary system can stop without taking a step. All enabled
monitor convergence conditions must be satisfied; ``solver.t_max`` remains an
independent upper limit. Normal finalisation and final-state output still run.
Regular sampling excludes the final completed ``t_max`` boundary, as for other
monitors; no extra final sample is forced.

Each measurement recomputes Hamiltonian fields at the current spin configuration
and time, then evaluates the ordinary LLG right-hand side with the solver's
per-site damping and effective gyromagnetic ratio, including its
``gilbert_prefactor`` setting. This costs an additional field evaluation and
reduction per sample. CUDA reduces on the device and transfers only the scalar
result. No integration-stage derivatives are reused and no noise is generated.

Thermal noise is excluded: this is a deterministic drift diagnostic, principally
for zero-temperature relaxation, not a finite-temperature equilibration test.
A small rate establishes stationarity, not that the state is a stable energy
minimum. For driven systems, set the minimum duration to cover relevant future
pulses; consecutive samples cannot rule out a later change in the drive.
