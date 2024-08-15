crystal-field
=============

The crystal field Hamiltonian for a single atomic site is defined as

.. math::
      \mathcal{H} = \sum_{l=2,4,6} A_l(J) \sum_{m=-l\dots l} B_{l,-m} (-1)^m d_{l}^{0m}(\theta)e^{-i m \phi}

with the definition

.. math::
    d_{l}^{0m}(\theta) = \sqrt{\frac{(l-m)!}{(l+m)!}}P_{l}^{m}(\cos\theta) = (-1)^m d_l^{0-m}(\theta)

and

.. math::
    A_2(J) &= J (J - 1/2) \alpha_J \\
    A_4(J) &= J (J - 1/2) (J - 1) (J - 3/2) \beta_J\\
    A_6(J) &= J (J - 1/2) (J - 1) (J - 3/2) (J - 2) (J - 5/2) \gamma_J

with Stevens factors :math:`\alpha_J`, :math:`\beta_J`, :math:`\gamma_J` and :math:`J` from Hund's rules.
:math:`B_{l,m}` is a complex crystal field coefficient.

The Hamiltonian is only implemented for :math:`l=2,4,6`.

.. note::

    This Hamiltonian can be equivalently written in terms of spherical harmonics as

    .. math::
        \mathcal{H} = \sum_{l=2,4,6} A_l(J) \sum_{m=-l\dots l} B_{l,-m} \sqrt{\frac{4\pi}{2l+1}}Y_{l,-m}(\theta, \phi).

    or tesseral harmonics as

    .. math::
        \mathcal{H} = \sum_{l=2,4,6} A_l(J) \sum_{m=-l\dots l} C_{l,m} Z_{l-m}(\theta, \phi)

    where the tesseral harmonics are defined as

    .. math::
        Z_{l,m}(\theta,\phi) = \begin{cases}
                \sqrt{2}\Re(Y_{l,m}) & m > 0 \\
                Y_{l,0} & m = 0\\
                \sqrt{2}\Im(Y_{l,m}) & m < 0.
         \end{cases}

    and the coefficients :math:`C_{l,m}` are

    .. math::
        C_{l,m} = \begin{cases}
                \sqrt{\frac{2\pi}{2l + 1}}B_{l,-m} & m \neq 0 \\
                \sqrt{\frac{4\pi}{2l + 1}}B_{l,0} & m = 0\\
         \end{cases}

The input for JAMS is :math:`J`, :math:`\alpha_J`, :math:`\beta_J`, :math:`\gamma_J` and :math:`B_{l,m}` for each
material or unit cell position.

Settings
########

.. describe:: energy_units (optional | string)

    Energy units of the crystal field coefficients in one of the JAMS supported units

.. describe:: energy_cutoff (required | float)

    Coefficients with an absolute value less than this (technically tesseral coefficient
    :math:`|C_{l,m}| < E_{\mathrm{cutoff}}`)
    will be set to zero. This setting is also used to check that the imaginary part of the
    energy is less than :math:`E_{\mathrm{cutoff}}` after conversion from complex crystal field coefficients
    :math:`B{l,m}` to tesseral coefficients :math:`C{l,m}`. If this check fails then JAMS will error and
    the input should be checked. Units for the cutoff are the same as :code:`energy_units` so the cutoff and the
    interpretation of a negligible energy should be with respect to these units.

.. describe:: crystal_field_spin_type (required | "up" or "down")

    The crystal field input file contains data for both spin up and spin down. This setting selects which data to use.
    The choice should be made based on the physics of the local moment and the filling of the f-shell.

.. describe:: crystal_field_coefficients (required | list)

    A list of the crystal field parameters for each material or unit cell position. Each list element
    is another list with the format: :code:`(material, J, alphaJ, betaJ, gammaJ, cf_param_filename)`, where
    :code:`material` can be a material name or unit cell position, and :code:`cf_param_filename` is a filename
    for the file which contains the crystal field coefficients :math:`B_{l,m}` for that material.

Crystal Field File Format
#########################

The crystal field input file should have columns of data in the format :code:`l m upRe upIm dnRe dnIm` which
are :math:`l`, :math:`m`, :math:`\Re(B_{l,m}^{\uparrow})`, :math:`\Im(B_{l,m}^{\uparrow})`,
:math:`\Re(B_{l,m}^{\downarrow})`, :math:`\Im(B_{l,m}^{\downarrow})` with the units given in the :code:`energy_units`
setting. Coefficients should only be given for :math:`l=0,2,4,6` and :math:`m = -l \dots l`. Any missing coefficients will
be set to zero.

Example
#######

.. code-block:: none

    hamiltonians = (
      {
        module = "crystal-field";
        debug = false;
        energy_units = "Kelvin"
        energy_cutoff = 1e-1;
        crystal_field_spin_type = "down";
        crystal_field_coefficients = (
          ("Tb", 6, -0.01010101, 0.00012244, -0.00000112, "Tb.CFparameters.dat"));
      }
    );