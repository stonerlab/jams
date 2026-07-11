neutron-scattering-no-lattice
=============================

Calculates the partial scattering cross section for systems without a Bravais lattice.
The ``sigma_*`` columns are reported as the magnetic double-differential cross
section in ``barn sr^-1 meV^-1 unitcell^-1``. The monitor assumes
``k_f / k_i = 1`` and uses material 0 for the single-material spin amplitude
``S = moment / (g_e mu_B)``.

.. math:: 
	  \frac{d^2\sigma}{d\Omega dE} &= \frac{\left(\gamma r_0\right)^2}{2\pi\hbar} \frac{k_1}{k_0} \sum_{\alpha\beta}\left( \delta_{\alpha\beta} - \tilde{Q}_{\alpha}\tilde{Q}_{\beta}\right) \\
  &\times f(\vec{Q})^2 \left\langle \overline{\hat{S}^{\alpha}(\vec{Q},\omega)} \hat{S}^{\beta}(\vec{Q},\omega) \right\rangle 

.. warning::
	This monitor current only supports a single material.

Optional settings
^^^^^^^^^^^^^^^^^

.. describe:: kvector


.. describe:: periodogram

The frequencies are calculated using `Welch’s method <https://en.wikipedia.org/wiki/Welch%27s_method>`_ of overlapping periodograms.

- **length** (int | 1000): number of outputs over which to calculate periodogram (i.e. output_steps x timesteps)
- **overlap** (int | 500): number of outputs to overlap in sequential periodogram

.. code-block:: none

  periodogram : {
    length = 1000;
    overlap = 500;
  };

.. describe:: form_factor

Form-factor ``q`` values are reciprocal-lattice Cartesian coordinates without
the ``2*pi`` factor. For lattice parameter ``a`` in Angstrom, the International
Tables argument is ``s = |Q|/(4*pi)`` with ``Q = 2*pi*q/a``, so
``s = |q|/(2*a)``.

.. code-block:: none

  form_factor = (
    {
      g  = [2.0, 0.0, 0.0, 0.0];
      j0 = (0.3972, 13.244, 0.6295, 4.903, -0.0314, 0.350, 0.0044);
    }
  );


.. describe:: polarizations

List of arrays of neutron polarisation vectors.

.. code-block:: none

  polarizations = (
    [0.0, 0.0, 1.0],
    [0.0, 0.0,-1.0]
  );
