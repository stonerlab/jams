monte-carlo-constrained-cpu
===========================

This solver uses the Constrained Monte Carlo algorithm
[`Phys. Rev. B 82, 054415 (2010) <https://doi.org/10.1103/PhysRevB.82.054415>`_] to calculate equilibrium
thermodynamics with classical (Rayleigh Jeans) statistics while constraining
the order parameter to a given angle. The length of the order parameter is
free to vary. Usually this solver is used in combination with the
:ref:`torque monitor <torque-monitor>` to calculate free energy barriers.

The constrained collective vector can be either the total magnetisation

.. math::
    \vec{M} = \sum_i \mu_{s,i} \vec{S}_i,

or a material-transformed order parameter

.. math::
    \vec{N} = \sum_i \mu_{s,i} \left(\mathbb{T}_i \cdot \vec{S}_i\right).

Here :math:`\mu_{s,i}` and :math:`\mathbb{T}_i` are the magnetic-moment
magnitude and spin transform matrix of spin :math:`i`, as defined for its
material (see :ref:`materials`). The sum includes every spin in the simulated
supercell. Use the magnetisation constraint for a conventional ferro- or
ferrimagnet when the direction of its physical total moment is required. Use
the material-transformed constraint, for example, to flip one sublattice and
constrain a Neel vector in an antiferromagnet or a transformed ferrimagnetic
order parameter. The material-transformed definition is the default for
backward compatibility.

Two spatial constraint modes are available. The default ``"global"`` mode
uses the sums above over every spin in the simulated supercell. The
``"spin_spiral"`` mode instead partitions the supercell into unit-cell planes
normal to one reciprocal-lattice direction. For each plane :math:`p`, it
constrains either

.. math::
    \vec{M}_p = \sum_{i\in p} \mu_{s,i}\vec{S}_i

or

.. math::
    \vec{N}_p = \sum_{i\in p} \mu_{s,i}
    \left(\mathbb{T}_i\cdot\vec{S}_i\right).

The magnitude of every constrained vector remains free. Individual unit-cell
vectors within a plane are not constrained and may fluctuate away from the
plane direction.

For an axis-aligned reciprocal wavevector with nonzero component :math:`q_d`,
the target direction of the plane at integer unit-cell coordinate :math:`n_d`
is

.. math::
    \hat{\vec{c}}_{n_d} =
    R_{\hat{\vec{a}}}\!\left(2\pi q_d n_d\right)\hat{\vec{c}}_0,

where :math:`\hat{\vec{a}}` is the Cartesian spin-rotation axis and

.. math::
    \hat{\vec{c}}_0 =
    (\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta).

The wavevector is specified in cycles per unit cell, so the factor
:math:`2\pi` is applied by the solver. Planes are grouped by their integer
unit-cell coordinate, not merely by target direction: planes separated by a
full turn remain independent constraints.

The same plane grouping is also available at :math:`Q=0`. In this case every
plane has target direction :math:`\hat{\vec{c}}_0`, but each plane remains a
separate constrained collective vector and compensation spins are still
selected from that plane. This is deliberately different from ``"global"``
mode, which constrains only the collective vector summed over the complete
supercell. Use the plane-constrained :math:`Q=0` form when a zero-wavevector
reference must have the same constrained degrees of freedom as finite-
:math:`Q` spin-spiral calculations.

This Monte Carlo solver moves **two** spins for every trial. We define One Monte
Carlo step as one trial move of every spin on average. Therefore `num_spins/2`
trial moves of pairs of spins are made for each Monte Carlo step.

In global mode both trial spins are selected globally as before. In
spin-spiral mode the first spin is selected globally and the second
(compensation) spin is selected uniformly from the other nonzero-moment spins
in the same plane. It may belong to any unit cell or material in that plane.

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

.. describe:: cmc_constraint_mode = "global"

Selects the spatial collective vectors whose directions are constrained.
Valid, case-insensitive values are ``"global"`` for one supercell-wide vector
and ``"spin_spiral"`` for one vector per unit-cell plane.

.. describe:: cmc_constraint_type = "material_transform"

Selects the collective vector whose direction is constrained. Valid values are
``"magnetisation"`` for :math:`\vec{M}` and ``"material_transform"`` for
:math:`\vec{N}`. Values are case-insensitive.

For example, constrain the physical total magnetisation with

.. code-block:: cfg

    cmc_constraint_type = "magnetisation";

or explicitly select the material transforms with

.. code-block:: cfg

    cmc_constraint_type = "material_transform";

.. describe:: cmc_spiral_wavevector

Required only when ``cmc_constraint_mode = "spin_spiral"``. A three-component
reciprocal-lattice vector in cycles per unit cell. Every component must be
finite and at most one component may be nonzero; oblique wavevectors are
rejected.

For a nonzero wavevector, the propagation direction is inferred from its
nonzero component as before. For a zero wavevector,
``cmc_spiral_propagation_direction`` is required so that the solver knows how
to construct the constraint planes. A zero wavevector is always commensurate.

If propagation direction :math:`d` is periodic and contains :math:`L_d` unit
cells, the spiral must satisfy

.. math::
    q_d L_d \in \mathbb{Z}.

The numerical distance from :math:`q_dL_d` to the nearest integer may not
exceed :math:`10^{-8}`. An arbitrary wavelength is allowed when the
propagation direction has open boundaries.

.. describe:: cmc_spiral_propagation_direction

An integer selecting the lattice direction normal to the constraint planes:
``0`` for :math:`a`, ``1`` for :math:`b`, or ``2`` for :math:`c`. This setting
is mandatory when ``cmc_spiral_wavevector = [0.0, 0.0, 0.0]``.

For a nonzero wavevector this setting is optional. If supplied, it must agree
with the direction inferred from the nonzero wavevector component.

.. describe:: cmc_spiral_axis

Required only when ``cmc_constraint_mode = "spin_spiral"``. A finite,
nonzero Cartesian vector defining the spin-space rotation axis
:math:`\hat{\vec{a}}`. It is normalised internally.

For example, a planar spiral making one turn over four periodic unit cells in
the :math:`a` direction is configured with

.. code-block:: cfg

    cmc_constraint_mode = "spin_spiral";
    cmc_constraint_type = "magnetisation";
    cmc_constraint_theta = 90.0;
    cmc_constraint_phi = 0.0;
    cmc_spiral_wavevector = [0.25, 0.0, 0.0];
    cmc_spiral_axis = [0.0, 0.0, 1.0];

Every constrained plane must contain at least two nonzero-moment spins so that
a compensation spin is available.

For example, the following configurations produce plane-constrained
:math:`Q=0` references normal to the :math:`a`, :math:`b`, and :math:`c`
directions, respectively:

.. code-block:: cfg

    cmc_constraint_mode = "spin_spiral";
    cmc_spiral_wavevector = [0.0, 0.0, 0.0];
    cmc_spiral_axis = [0.0, 0.0, 1.0];
    cmc_spiral_propagation_direction = 0; // a planes

.. code-block:: cfg

    cmc_constraint_mode = "spin_spiral";
    cmc_spiral_wavevector = [0.0, 0.0, 0.0];
    cmc_spiral_axis = [0.0, 0.0, 1.0];
    cmc_spiral_propagation_direction = 1; // b planes

.. code-block:: cfg

    cmc_constraint_mode = "spin_spiral";
    cmc_spiral_wavevector = [0.0, 0.0, 0.0];
    cmc_spiral_axis = [0.0, 0.0, 1.0];
    cmc_spiral_propagation_direction = 2; // c planes

The spiral axis is still required, finite, nonzero, and normalised internally
at :math:`Q=0`, even though it has no numerical effect when every phase is
zero.

.. describe:: cmc_constraint_tolerance = 1e-6

Maximum permitted angular separation, in degrees, between the requested
constraint direction and the measured collective-vector direction during the
solver's periodic constraint validation. The value must be finite and satisfy
``0 < cmc_constraint_tolerance <= 180``.

This setting controls only the threshold at which validation stops the solver;
it does not alter or relax the constrained Monte Carlo trial moves themselves.
The geometric direction comparison is independent of the azimuthal branch cut
and remains well-defined at the polar directions.

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
terminal every :code:`output_write_steps` steps.

.. describe:: move_fraction_uniform = 0.0

Fraction between 0 and 1 of trial moves which move a spin to a uniform
random angle on the sphere.

.. math::
	  (S_x, S_y, S_z) \rightarrow (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

.. describe:: move_fraction_angle = 1.0

Fraction between 0 and 1 of trial moves which move a spin by a limited angle.
The size of the angle is controlled by  :code:`move_angle_sigma`.

.. describe:: move_angle_sigma = 0.5

The initial dimensionless size :math:`\sigma` in
:code:`move_fraction_angle`. It controls the angular extent of the proposal
but is not itself an angle in degrees or radians.

.. math::
	  (S_x, S_y, S_z) \rightarrow (S_x, S_y, S_z) + \sigma(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta) \quad \mathrm{where}\quad \theta\sim[0,\pi],\phi\sim[0,2\pi)

Adaptive angle moves
""""""""""""""""""""

Angle-move adaptation is disabled by default. Enable it with a nested block:

.. code-block:: cfg

    move_angle_sigma = 0.01;
    move_angle_adaptation = {
      enabled = true;
      target_acceptance = 0.10;
      interval_steps = 100;
      gain = 0.5;
      min_sigma = 1.0e-6;
      max_sigma = 0.1;
      burn_in_steps = 10000;
    };

After each adaptation interval the solver uses only angle-move trials from
that interval and updates in logarithmic space:

.. math::

    \log\sigma_{n+1} = \operatorname{clamp}\!\left(
      \log\sigma_n + \gamma(a_n-a_{\mathrm{target}}),
      \log\sigma_{\min}, \log\sigma_{\max}\right).

Uniform and reflection moves do not contribute to :math:`a_n`. If no angle
moves were attempted in an interval, the update is skipped. Adaptation
intervals are independent of ``output_write_steps``.

.. describe:: move_angle_adaptation.enabled = false

Enables adaptive tuning when ``true``. The other settings below are required
when adaptation is enabled.

.. describe:: move_angle_adaptation.target_acceptance

Dimensionless target angle-move acceptance fraction. It must be finite and
satisfy ``0 < target_acceptance < 1``.

.. describe:: move_angle_adaptation.interval_steps

Positive integer number of Monte Carlo steps between updates.

.. describe:: move_angle_adaptation.gain

Finite positive dimensionless gain :math:`\gamma` in the update equation.

.. describe:: move_angle_adaptation.min_sigma

Finite positive dimensionless lower bound for :math:`\sigma`.

.. describe:: move_angle_adaptation.max_sigma

Finite positive dimensionless upper bound for :math:`\sigma`. The bounds and
initial value must satisfy
``min_sigma <= move_angle_sigma <= max_sigma``.

.. describe:: move_angle_adaptation.burn_in_steps

Positive integer burn-in duration in Monte Carlo steps, no greater than
``max_steps``. At nonzero temperature this setting is mandatory: adaptation
uses only the burn-in samples and the proposal width is frozen before
production sampling. A final partial interval is updated at the burn-in
boundary before freezing.

At zero temperature ``burn_in_steps`` is optional. If omitted, adaptation
continues for the complete constrained minimisation; if present, it freezes at
the specified boundary. Unrestricted zero-temperature adaptation is intended
for minimisation, not equilibrium sampling. A runtime guard stops a run if its
temperature changes from zero to nonzero while unrestricted adaptation is
active.

The solver logs every update, skipped interval, reached bound, and frozen
production value. Adaptive counters and the current :math:`\sigma` are not
serialised by the existing restart mechanism. A restarted process therefore
begins again from the configured ``move_angle_sigma``; only the restored spin
configuration is retained.

.. describe:: move_fraction_reflection = 0.0

Fraction between 0 and 1 of trial moves reflect a spin.

.. math::
	  (S_x, S_y, S_z) \rightarrow (-S_x, -S_y, -S_z)

.. warning::
    This trial move is non-ergodic for Heisenberg spins and **must** be used
    in combination with other types of trial move.
