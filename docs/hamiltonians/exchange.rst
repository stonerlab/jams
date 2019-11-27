exchange
========

Heisenberg bilinear exchange on a Bravis lattice

.. math:: 
      \mathcal{H} = -\tfrac{1}{2}\sum_{ij} \vec{S}_{i} \cdot \mathbb{J}_{ij} \cdot \vec{S}_j

The exchange tensors :math:`\mathbb{J}_{ij}` are specified between materials
or atom positions within the unit cell with an interaction vector
:math:`\vec{r}_{ij}`. By default the interaction vector is in Cartesian
coordinates and JAMS will use the detected symmetry operations of the
unit cell to generate symmetric vectors.

Specified interactions are not assumed to be reciprocal between
materials/unitcell positions, so  if :math:`A` and :math:`B` are materials
:math:`\mathbb{J}_{AB}` and :math:`\mathbb{J}_{BA}` must both be specified.


Settings
########

.. describe:: exc_file

    Name of file containing exchange interaction data (format specified below)


.. describe:: interactions

    Inline description of exchange parameters.

    Format is one of:

    - :code:`("MaterialA", "MaterialB", [rx, ry, rz], Jij)`
    - :code:`("MaterialA", "MaterialB", [rx, ry, rz], Jij_xx, Jij_xy, Jij_xz, Jij_yx, Jij_yy, Jij_yz, Jij_zx, Jij_zy, Jij_zz)`
    - :code:`(1, 2, [rx, ry, rz], Jij)`


.. code-block:: none

    interactions = (
        ("A", "A", [0.5, 0.5, 0.5], 3.5e-21)
    );

.. warning::

    The data structure used to store the config means that if too many interactions are specified in the :code:`interactions` setting then the memory usage will be very large. In this case the external :code:`exc_file` should be used instead.

.. describe:: coordinate_format = "cartesian"

    "cartesian" or "fractional"

    Coordinate system for the interaction vectors rij.

.. describe:: symops = true

    If true then the symmetry operations of the crystal will be applied to each interaction vector.

.. describe:: print_unfolded = false

    If true then the input interactions are printed out to file after any symmetry operations have been applied.

.. describe:: energy_cutoff = 1e-26

    Remove any exchange interactions where abs(Jij) < energy_cutoff.

.. describe:: radius_cutoff = 100.0

    Remove any exchange interactions where norm(rij) > radius_cutoff (in units of lattice parameters).

.. describe:: distance_tolerance = jams::defaults::lattice_tolerance

    Tolerance to use for floating point comparisons of distances for rij.

