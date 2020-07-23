monte-carlo-constrained-cpu
===========================

This solver uses the Constrained Monte Carlo algorithm
[`Phys. Rev. B 82, 054415 (2010) <https://doi.org/10.1103/PhysRevB.82.054415>`_] to calculate equilibrium
thermodynamics with classical (Rayleigh Jeans) statistics while constraining
the order parameter to a given angle. The length of the order parameter is
free to vary. Usually this solver is used in combination with the
:ref:`torque monitor <torque-monitor>` to calculate free energy barriers.

The constraint angles are applied to the order parameter vector

.. math::
    \vec{N} = \sum_{i} \mu_{s,i} \left( \mathbb{T}_{i} \cdot \vec{S}_{i} \right)

where :math:`\mu_{s,i}` and :math:`\mathbb{T}_{i}` are the magnetic moment and
the spin transform matrix of the :math:`i`-th spins as defined for each material
(see :ref:`materials`). The transformation matrix allows Constrained Monte Carlo
to be used for example with ferrimagnets and antiferrimagnets by use of the Neel
vector rather than the magnetisation.

This Monte Carlo solver moves **two** spins for every trial. We define One Monte
Carlo step as one trial move of every spin on average. Therefore `num_spins/2`
trial moves of pairs of spins are made for each Monte Carlo step.

Trial spins are always chosen at random, both for the initial and second
(compensation) spin in the pair. No consideration is given for whether the
second spin is from the same material or unit cell position.

.. note::
    The solver will periodically check the constraint is being correctly maintained.
    If it is not constrained correctly JAMS will quit with an error. This indicates
    there is either a problem with the input configuration or an unexpected bug in
    JAMS.

Required settings
^^^^^^^^^^^^^^^^^

.. describe:: max_steps

Maximum number of Monte Carlo steps to solve.

.. describe:: cmc_constraint_theta

Polar (from :math:`z`-axis) constraint angle in degrees.

.. describe:: cmc_constraint_phi

Azimuthal (in :math:`xy`-plane)  constraint angle in degrees.

Optional settings
^^^^^^^^^^^^^^^^^

.. describe:: min_steps = 0

Minimum number of Monte Carlo steps to solve (in case a monitor can stop
the solver due to a convergence criterion).

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

Statistics about how many moves were accepted of each type are printed to the
terminal every :option:`output_write_steps` steps.

.. describe:: move_fraction_uniform = 0.0

Fraction between 0 and 1 of trial moves which move a spin to a uniform
random angle on the sphere.

.. math::
	  (S_x, S_y, S_z) \rightarrow (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

.. describe:: move_fraction_angle = 1.0

Fraction between 0 and 1 of trial moves which move a spin by a limited angle.
The size of the angle is controlled by  :option:`move_angle_sigma`.

.. describe:: move_angle_sigma = 0.5

The size of :math:`\sigma` in :option:`move_fraction_angle`.

.. math::
	  (S_x, S_y, S_z) \rightarrow (S_x, S_y, S_z) + \sigma(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

.. describe:: move_fraction_reflection = 0.0

Fraction between 0 and 1 of trial moves reflect a spin.

.. math::
	  (S_x, S_y, S_z) \rightarrow (-S_x, -S_y, -S_z)

.. warning::
    This trial move is non-ergodic for Heisenberg spins and **must** be used
    in combination with other types of trial move.
