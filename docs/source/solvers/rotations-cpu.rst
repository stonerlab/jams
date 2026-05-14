.. _rotations-cpu:

rotations-cpu
=============

``rotations-cpu`` rotates the spin state over a grid of spherical angles. Legacy
``num_theta`` and ``num_phi`` settings still scan the full sphere with inclusive
endpoints:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      num_theta = 36;
      num_phi = 72;
    };

The ``theta`` and ``phi`` settings can now restrict either angle to a constant,
a range, or an explicit list. Angles are specified in degrees.

Constant theta with a phi sweep:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      theta = { value_deg = 90.0; };
      phi = { start_deg = 0.0; stop_deg = 360.0; count = 72; endpoint = false; };
    };

Constant phi with a theta sweep:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      theta = { start_deg = 0.0; stop_deg = 180.0; count = 36; };
      phi = { value_deg = 45.0; };
    };

Explicit angle values:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      theta = { values_deg = [ 30.0, 60.0, 90.0 ]; };
      phi = { values_deg = [ 0.0, 90.0, 180.0, 270.0 ]; };
    };

Scalar shorthand is also accepted for constant angles:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      theta = 90.0;
      phi = { start_deg = 0.0; stop_deg = 180.0; count = 19; };
    };
