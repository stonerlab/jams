exchange
========

Heisenberg bilinear exchange on a Bravis lattice

.. math:: 
      \mathcal{H} = -\tfrac{1}{2}\sum_{ij} \vec{S}_{i} \overline{\overline{J}}_{ij} \vec{S}_j




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

