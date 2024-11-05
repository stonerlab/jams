neutron-scattering-no-lattice
=============================

Calculates the partial scattering cross section for systems without a Bravis lattice.

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