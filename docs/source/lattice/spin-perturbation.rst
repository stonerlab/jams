Small random spin perturbations
==============================

The ``spin-perturbation`` initialiser adds independent small random tilts to
the existing spins. It can break an exactly stationary but unstable state,
such as spins antiparallel to an applied field, before LLG dynamics begins.

.. code-block:: none

   initializer = {
     module = "spin-perturbation";
     amplitude = 1.0e-3;
     seed = 1234;
   };

Both settings are required. ``amplitude`` is finite, non-negative and
dimensionless. ``seed`` is an integer from 0 to 2147483647. A local PCG random
generator makes the perturbation reproducible within the same build for the
same input spin configuration, seed and site order, without advancing the
simulation or thermostat random generators. Bitwise agreement across builds
or platforms is not guaranteed: compiler settings, floating-point arithmetic
and standard-library implementations can change the results.
The perturbation is independent of ``sim.seed``. Zero amplitude leaves the
spins exactly unchanged. Vacancies (zero magnetic moment) are skipped without
drawing random numbers. All other spins must have finite, non-zero length.

For each spin :math:`\mathbf S=\ell\mathbf n`, with :math:`|\mathbf n|=1`,
draw a uniform random unit vector :math:`\mathbf u` and apply

.. math::

   \mathbf p = \mathbf u-(\mathbf u\cdot\mathbf n)\mathbf n,
   \qquad
   \mathbf S' = \ell\frac{\mathbf n+\epsilon\mathbf p}
     {\sqrt{1+\epsilon^2|\mathbf p|^2}}.

The transverse projection uses each spin's local direction and preserves its
original length after normalisation. The tilt satisfies
:math:`\theta\leq\arctan\epsilon`; an amplitude of ``1e-3`` gives a maximum
tilt of approximately 0.057 degrees. This is neither a fixed-angle perturbation
nor a uniform distribution over a cone. No material or spatial filtering is
applied.

Initialisers execute in list order. Put the perturbation after any initialiser
that sets the spin texture or loads a saved state, for example:

.. code-block:: none

   initializer = (
     {module = "h5"; spins = "initial.h5";},
     {module = "spin-perturbation"; amplitude = 1.0e-3; seed = 1234;}
   );

The initialiser runs once on the CPU before solver creation and works with
CPU and CUDA solvers. Physics modules that subsequently constrain or overwrite
spins can modify the perturbation.

For relaxation, use positive damping. With the ``spin-rate`` monitor, choose a
threshold below the perturbed initial rate or allow growth with ``solver.t_min``.
A perturbation seeds motion but does not establish stability or guarantee
escape within a specified time. It is not a thermal noise model.
