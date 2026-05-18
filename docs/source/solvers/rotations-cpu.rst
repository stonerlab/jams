.. _rotations-cpu:

rotations-cpu
=============

``rotations-cpu`` scans static spin configurations over spherical angles. It is
not a time-integration solver. At each solver iteration it restores the initial
spin state, applies the current angle sample, and then lets the configured
monitors evaluate that state.

Angles use the usual spherical convention: ``theta`` is the polar angle from the
positive z axis, and ``phi`` is the azimuthal angle in the x-y plane from the
positive x axis.

Behaviour
---------

The solver has two target modes:

``rotate_all_spins = true``
  The default. Each angle sample applies one common rotation matrix to every
  spin in the initial state.

``rotate_all_spins = false``
  Each basis spin target is scanned independently. The solver restores the
  initial state, assigns the current target spin to the sampled spherical
  direction, and reports the target as ``spin_index`` in monitor coordinate
  output.

The monitor coordinate columns are:

``rotation_step``
  The solver iteration index.

``spin_index``
  Only present when ``rotate_all_spins = false``.

``phi_deg`` and ``theta_deg``
  The current angle sample in degrees.

Legacy full-sphere scan
-----------------------

Legacy ``num_theta`` and ``num_phi`` settings still scan the full sphere with
inclusive endpoints:

.. code-block:: none

    solver = {
      module = "rotations-cpu";
      num_theta = 36;
      num_phi = 72;
    };

This is equivalent to a theta range from 0 to 180 degrees and a phi range from 0
to 360 degrees. The endpoint is included in both directions for backwards
compatibility.

Angle specifications
--------------------

The ``theta`` and ``phi`` settings can restrict either angle to a constant, a
range, or an explicit list. All new angle settings are specified in degrees and
are converted to radians internally.

A scalar setting is shorthand for a constant angle:

.. code-block:: none

    theta = 90.0;

The grouped constant form is more explicit:

.. code-block:: none

    theta = { value_deg = 90.0; };

A range uses ``start_deg``, ``stop_deg`` and ``count``:

.. code-block:: none

    theta = { start_deg = 0.0; stop_deg = 180.0; count = 37; };

By default, a range includes ``stop_deg``. Set ``endpoint = false`` to omit the
upper endpoint:

.. code-block:: none

    phi = { start_deg = 0.0; stop_deg = 360.0; count = 72; endpoint = false; };

If ``count`` is omitted from a theta or phi range, the corresponding
``num_theta`` or ``num_phi`` value is used. If ``count`` is 1, the range contains
only ``start_deg``.

Explicit values can be supplied as a list:

.. code-block:: none

    phi = { values_deg = [ 0.0, 90.0, 180.0, 270.0 ]; };

The ``value_deg``, ``values_deg`` and range forms are mutually exclusive for a
single angle.

Examples
--------

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
