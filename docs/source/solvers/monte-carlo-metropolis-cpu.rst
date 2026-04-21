monte-carlo-metropolis-cpu
==========================

This solver uses the Metropolis Monte Carlo algorithm to calculate equilibrium
thermodynamics with classical (Rayleigh Jeans) statistics.

One Monte Carlo step is defined as a trial move having been attempted for
every spin on average. On average is because spins can be selected at random
so there is no guarantee that every spin has a trial move, only that
`num_spins` trial moves occur.

Required settings
^^^^^^^^^^^^^^^^^

.. describe:: max_steps

Maximum number of Monte Carlo steps to solve.

Optional settings
^^^^^^^^^^^^^^^^^

.. describe:: min_steps = 0

Minimum number of Monte Carlo steps to solve (in case a monitor can stop
the solver due to a convergence criterion).

.. describe:: select_spins_by_random = true

Toggle whether spins are selected randomly for trial moves or if the spins
are selected in-order. On some systems/compilers the in-order selection
has been seen to be 4x faster. Both methods should converge to the correct
thermodynamic solution given enough steps.

.. describe:: output_write_steps = 1000

Number of Monte Carlo steps between outputting trial move statistics to the
terminal.

Trial Moves
"""""""""""

Different types of trial spin moves can be used. Selecting different types or
combination of types can give a much faster convergence to equilibrium.
The total move fraction should add to 1 (JAMS will normalise anyway).

The move to use for a given Monte Carlo step is chosen randomly but the same
move is used for every trial move within one step.

Statistics about how many moves were accepted of each type are written to
``monte_carlo_stats.tsv`` every ``output_write_steps`` steps.

.. describe:: move_fraction_uniform = 0.0

Fraction between 0 and 1 of trial moves which move a spin to a uniform
random angle on the sphere.

.. math::
	  (S_x, S_y, S_z) \rightarrow (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

.. describe:: move_fraction_angle = 1.0

Fraction between 0 and 1 of trial moves which move a spin by a limited angle.
The size of the angle is controlled by  `move_angle_sigma`.

.. describe:: move_angle_sigma = 0.5

The size of :math:`\sigma` in `move_fraction_angle`.

.. math::
	  (S_x, S_y, S_z) \rightarrow (S_x, S_y, S_z) + \sigma(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

.. describe:: move_fraction_reflection = 0.0

Fraction between 0 and 1 of trial moves reflect a spin.

.. math::
	  (S_x, S_y, S_z) \rightarrow (-S_x, -S_y, -S_z)

.. warning::
    This trial move is non-ergodic for Heisenberg spins and **must** be used
    in combination with other types of trial move.
