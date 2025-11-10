exchange-neartree
=================

Heisenberg bilinear exchange based on spherical distance

.. math::
      \mathcal{H} = -\tfrac{1}{2}\sum_{ij} J(r_{ij}) \mathbf{S}_{i} \cdot \mathbf{S}_j

The exchange :math:`J(r_{ij})` is specified between materials. Neighbours
with the given material types at distance :math:`r_{ij}` will be given the exchange value.
Note that this assumes spherical shells and is most appropriate for simple lattices
(e.g. SC, BCC, FCC) with high symmetry. For systems with more complex symmetry it may
be the case that exchange interactions have the same distance but belong to different
stars of :math:`\mathbf{r}` which are not linked by symmetry. No checks are made for violations
of such symmetry.

.. note::
    Specified interactions are assumed to be reciprocal between
    materials/unitcell positions, so  if :math:`A` and :math:`B` are materials only one of
    :math:`J_{AB}` and :math:`J_{BA}` should be specified.


Settings
########

.. describe:: interactions (required | list)

    Description of exchange parameters.

    Format is :code:`("MaterialA", "MaterialB", r_ij, J_ij)`


.. code-block:: none

    interactions = (
        ("A", "A", 1.0, 3.5e-21)
    );

.. describe:: energy_units (optional | string | "joules")

    Energy units of the exchange coefficients in one of the JAMS supported units.

.. describe:: distance_units (optional | string | "lattice_constants")

    Distance units of :math:`r_{ij}` in one of the JAMS supported units.

.. describe:: energy_cutoff (optional | float | 1e-26)

    Remove any exchange interactions where abs(Jij) < energy_cutoff.

.. describe:: shell_width (optional | float | 1e-3)

    Width of the shell around the radius :math:`r_{ij}` to detect neighbours to account for
    small precision errors (in units of :code:`distance_units`).


Example
#######

Ni-Fe on an FCC lattice.

.. code-block:: none

    hamiltonians = (
      {
        module = "exchange-neartree";
        energy_units = "meV";
        distance_units = "lattice_constants";
        shell_width = 1e-3;
        interactions = (
           // Shell 1 (r = 1/sqrt(2), 12 nbrs)
           ("Ni", "Ni", 0.70710678, 17.0),
           ("Ni", "Fe", 0.70710678, 31.0),
           ("Fe", "Fe", 0.70710678, 44.0)
        );
      }
    );