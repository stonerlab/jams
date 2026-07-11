neutron-scattering
==================

Calculates the magnetic neutron-scattering cross section on a reciprocal-space
path. By default the monitor samples spatial Fourier components on the FFT grid
defined by the simulation supercell.

The ``sigma_*`` columns are reported as the magnetic double-differential cross
section in ``barn sr^-1 meV^-1 unitcell^-1``. The monitor assumes
``k_f / k_i = 1`` and converts the configured material moments to spin
amplitudes using ``S = moment / (g_e mu_B)``.

When magnetic form factors are configured, the monitor treats ``q.xyz`` as
reciprocal-lattice Cartesian coordinates without the ``2*pi`` factor. For a
lattice parameter ``a`` in Angstrom, the International Tables argument is
``s = |Q| / (4*pi)`` with ``Q = 2*pi*q.xyz/a``, so ``s = |q.xyz|/(2*a)``.

Exact-Q direct summation
^^^^^^^^^^^^^^^^^^^^^^^^

Set ``direct_sum.enabled = true`` to evaluate the spatial Fourier transform by
direct summation at the requested reciprocal-space points. The temporal
transform is unchanged: the stored ``S(Q,t)`` series is still transformed in
frequency using the configured periodogram estimator.

.. code-block:: none

  monitors = (
    {
      module = "neutron-scattering";
      output_steps = 1;

      direct_sum : {
        enabled = true;
        backend = "auto";   // auto | cpu | cuda
        hkl_path = (
          [0.0,  0.0, 0.0],
          [0.37, 0.0, 0.0],
          [0.37, 0.5, 0.0]
        );
        points_per_segment = [101, 51];

        window : {
          x : { origin = 20.0; width = 30.0; };
          z : { origin =  5.0; width = 10.0; };
        };
      };

      compute_periodogram : {
        length = 1000;
        overlap = 500;
      };
    }
  );

``direct_sum.hkl_path`` is interpolated exactly and is not clamped to FFT grid
points. ``points_per_segment`` is endpoint-inclusive; shared nodes between
consecutive line segments are emitted once. A scalar value applies to every
segment, while a list must contain one value for each line segment. A one-point
``hkl_path`` is valid and does not require ``points_per_segment``.

``direct_sum.window`` is optional. The ``x``, ``y``, and ``z`` entries are
independent Cartesian windows, so a single entry such as ``z`` is valid. Each
axis uses ``origin`` as the window center and ``width`` as the full Cartesian
extent. The monitor applies the default JAMS window function along each enabled
axis and excludes spins outside the requested extent before summing.
