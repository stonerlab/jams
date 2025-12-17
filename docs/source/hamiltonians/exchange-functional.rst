exchange-functional
===================

Heisenberg bilinear exchange with a functional form based on type and interaction vector

.. math::
      \mathcal{H} = -\tfrac{1}{2}\sum_{ij} J_{a,b}(\mathbf{r}_{ij}) \mathbf{S}_{a,i} \cdot \mathbf{S}_{b,j}

The exchange function :math:`J_{a,b}(\mathbf{r}_{ij})` is specified between materials :math:`a` and :math:`b`.
Neighbours with the given material types will be given the exchange value calculate from :math:`J_{a,b}(\mathbf{r}_{ij})`.

.. note::
    Specified interactions are assumed to be reciprocal between
    materials, so  if :math:`A` and :math:`B` are materials only one of
    :math:`J_{AB}(\mathbf{r}_{ij})` and :math:`J_{BA}(\mathbf{r}_{ij})` should be specified. Moreover
    only one functional can be specified per material or material pair.

Functionals
###########

The available functionals are:

.. describe:: step

A step function where :math:`J_0` is constant within :math:`r_0` and
zero outside.

.. math::
    J(\mathbf{r}_{ij}) =
    \begin{cases}
        J_{0}, & |r_{ij}| \le r_{\mathrm{cut}}, \\
        0, & |r_{ij}| > r_{\mathrm{cut}} .
    \end{cases}

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0
        ("A", "A", "step", 1.0, 20.0)
    );

.. describe:: exp

Exponentially decaying function with a linear shift :math:`r_0` and
decay constant :math:`\sigma`.

.. math::
    J(\mathbf{r}_{ij}) = J_0 \operatorname{exp}\left( -\frac{|r_{ij}|-r_0}{\sigma} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "exp", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: gaussian

Gaussian function centered on :math:`r_0` with width :math:`\sigma`.

.. math::
    J(\mathbf{r}_{ij}) = J_0 \operatorname{exp}\left( -\frac{(|r_{ij}|-r_0)^2}{2\sigma^2} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "gaussian", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: kaneyoshi

Function used in Kaneyoshi's papers on amorphous magnets.

.. math::
    J(\mathbf{r}_{ij}) = J_0 (|r_{ij}|-r_0)^2\operatorname{exp}\left( -\frac{(|r_{ij}|-r_0)^2}{2\sigma^2} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "kaneyoshi", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: rkky

RKKY type interaction with oscillating sign. Defined with a shift :math:`r_0` and
wavenumber :math:`k`.

.. math::
    J(\mathbf{r}_{ij}) = -J_0 \frac{2 k (|r_{ij}|-r_0) \cos(2 k (|r_{ij}|-r_0)) - \sin(2 k (|r_{ij}|-r_0)}{(2 k (|r_{ij}| - r_0))^4}

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, wavenumber
        ("A", "A", "rkky", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: gaussian_multi

Multiple (three) gaussian functions with independent centers, widths and amplitudes.

.. math::
    J(\mathbf{r}_{ij}) = J_0 \operatorname{exp}\left( -\frac{(|r_{ij}|-r_0)^2}{2\sigma_0^2} \right) + J_1 \operatorname{exp}\left( -\frac{(|r_{ij}|-r_1)^2}{2\sigma_1^2} \right) + J_2 \operatorname{exp}\left( -\frac{(|r_{ij}|-r_2)^2}{2\sigma_2^2} \right)


.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma0, J1, r1, sigma1, J2, r2, sigma2
        ("A", "A", "gaussian_multi", 4.0, 20.0, 1.0, 1.0, -10.0, 2.0, 1.0, 5.0, 3.0, 1.0)
    );

.. describe:: c3z

Three fold rotationally symmetric function in the x-y plane. See `arXiv:2206.05264 <https://arxiv.org/abs/2206.05264>`_ and
`Nano Lett. 23, 6088 (2023) <https://dx.doi.org/10.1021/acs.nanolett.3c01529>`_.


.. math::
   J_{ij}
   =
   J_{0}\,\exp\left(-\lvert r_{ij} - d_{0} \rvert / l_{0}\right)
   + J_{1}^{\mathrm{s}}\,\exp\left(-\lvert r_{ij} - r_{*} \rvert / l_{1}^{\mathrm{s}}\right)
     \sum_{a=1}^{3} \sin\left(\mathbf{q}_{a}^{\mathrm{s}} \cdot \mathbf{r}_{\parallel}\right)
   + J_{1}^{\mathrm{c}}\,\exp\left(-\lvert r_{ij} - r_{*} \rvert / l_{1}^{\mathrm{c}}\right)
     \sum_{a=1}^{3} \cos\left(\mathbf{q}_{a}^{\mathrm{c}} \cdot \mathbf{r}_{\parallel}\right).

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, qs1, qc1, J0, J1s, J1c, d0, l0, l1s, l1c, r*
        ("A", "B", "c3z", 10.0, [0.7, 0.0, 0.0], [1.73, 1.0, 0.0], -0.1, -0.5, 0.1, 6.7, 0.1, 0.3, 0.6, 7.3),
    );

.. note::
    Only :math:`\mathbf{q}_{1}^{s,c}` are specified and :math:`\mathbf{q}_{2,3}^{s,c}` are calculated from the C3z symmetry.

Settings
########

.. describe:: interactions (required | list)

    Description of exchange parameters.

    Format is :code:`("MaterialA", "MaterialB", "functional_name", r_cutoff, functional_parameters...)`


.. describe:: energy_units (optional | string | "joules")

    Energy units of the exchange coefficients in one of the JAMS supported units.

.. describe:: distance_units (optional | string | "lattice_constants")

    Distance units of :math:`r_{ij}` in one of the JAMS supported units.

.. describe:: output_functionals (optional | bool | false)

    Output functionals to text files with columns of radius_nm and exchange_meV.


Example
#######

.. code-block:: none

    hamiltonians = (
      {
        module = "exchange-functional";
        energy_units = "meV";
        distance_units = "lattice_constants";
        interactions = (
            // type_i, type_j, functional, r_cutoff, J0, r0, sigma
            ("A", "A", "exp", 3.0, 20.0, 1.0, 1.0)
        );
      }
    );