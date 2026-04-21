rotations-cpu
=============

``rotations-cpu`` scans spin orientations on a grid of polar and azimuthal
angles and writes the corresponding Hamiltonian energies to ``ang_eng.tsv``.
It is a utility solver for mapping anisotropy and energy landscapes rather
than a time integrator.

Configuration
-------------

.. code-block:: cfg

   solver = {
     module = "rotations-cpu";
     rotate_all_spins = true;
     num_theta = 36;
     num_phi = 72;
   };

Optional settings
-----------------

``rotate_all_spins``
  If ``true``, rotate the whole spin configuration as a rigid body.
  If ``false``, rotate one basis spin at a time and write one file per basis
  site. Default: ``true``.

``num_theta``
  Number of polar-angle samples between :math:`0` and :math:`\pi`.
  Default: ``36``.

``num_phi``
  Number of azimuthal-angle samples between :math:`0` and :math:`2\pi`.
  Default: ``72``.

Output
------

When ``rotate_all_spins = true`` the solver writes ``ang_eng.tsv`` with the
columns

1. ``phi_deg``
2. ``theta_deg``
3. one energy column per Hamiltonian term

When ``rotate_all_spins = false`` the same data is written separately for each
basis spin.
