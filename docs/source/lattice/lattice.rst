.. _lattice:
Lattice
=======

.. describe:: size = [A, B, C]

    Number of unit cells in the a,b,c unit cell directions making up the simulation super cell.

.. describe:: periodic = [true, true, true]

    Apply periodic boundaries along the a, b, c unit cell vector directions.

Rotating the system
###################

.. describe:: global_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    Rotate the whole system using this rotation matrix.

.. describe:: orientation_axis

    Cartesian vector to orient to using either :code:`orientation_lattice_vector` or :code:`orientation_cartesian_vector`.

.. note::

    If both orientation and global_rotation are given then the orientation will be applied first and then global_rotation.

Adding impurities
#################

.. describe:: inpurities

.. describe:: impurities_seed (optional)

    Seed for generating random impurities. Using the same seed should give the same configuration of impurities.

.. code-block:: none

    inpurities = (
        ("MaterialA", "MaterialB", 0.5)
    );