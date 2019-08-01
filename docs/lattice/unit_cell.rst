Unit cell
=========


.. describe:: parameter

    Lattice parameter in meters.

.. describe:: basis

    Matrix of lattice vectors in the format:

.. code-block:: none

    basis = (         
     [a1, b1, c1],
     [a2, b2, c2],
     [a3, b3, c3]);

.. describe:: positions

    Position of atoms in the unit cell (fractional coordinates by default).

.. describe:: coordinate_format = "fractional"

    "fractional" or "cartesian"

    Coordinate format for the :code:`positions` setting.

Example
#######

.. code-block:: none

    unitcell: {
        parameter = 0.3e-9;

        basis = (         
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]);

       positions = (
        ("A", [0.00, 0.00, 0.00]),
        ("A", [0.50, 0.50, 0.50]));
 };
