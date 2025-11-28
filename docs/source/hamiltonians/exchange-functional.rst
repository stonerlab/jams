exchange-functional
===================

Heisenberg bilinear exchange with a functional form based on type and distance

.. math::
      \mathcal{H} = -\tfrac{1}{2}\sum_{ij} J_{a,b}(r_{ij}) \mathbf{S}_{a,i} \cdot \mathbf{S}_{b,j}

The exchange function :math:`J_{a,b}(r_{ij})` is specified between materials :math:`a` and :math:`b`.
Neighbours with the given material types at distance :math:`r_{ij}` will be given the exchange value.
Note that this assumes spherical shells and is most appropriate for simple lattices
(e.g. SC, BCC, FCC) with high symmetry. For systems with more complex symmetry it may
be the case that exchange interactions have the same distance but belong to different
stars of :math:`\mathbf{r}` which are not linked by symmetry. No checks are made for violations
of such symmetry.

.. note::
    Specified interactions are assumed to be reciprocal between
    materials, so  if :math:`A` and :math:`B` are materials only one of
    :math:`J_{AB}(r)` and :math:`J_{BA}(r)` should be specified. Moreover
    only one functional can be specified per material or material pair.

Functionals
###########

The available functionals are:

.. describe:: step

A step function where :math:`J_0` is constant within :math:`r_0` and
zero outside.

.. math::
    J(r) =
    \begin{cases}
        J_{0}, & r \le r_{\mathrm{cut}}, \\
        0, & r > r_{\mathrm{cut}} .
    \end{cases}

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0
        ("A", "A", "step", 1.0, 20.0)
    );

.. describe:: exp

Exponentially decaying function with a linear shift :math:`r_0` and
decay constant :math:`sigma`.

.. math::
    J(r) = J_0 \operatorname{exp}\left( -\frac{r-r_0}{\sigma} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "exp", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: gaussian

Gaussian function centered on :math:`r_0` with width :math:`sigma`.

.. math::
    J(r) = J_0 \operatorname{exp}\left( -\frac{(r-r_0)^2}{2\sigma^2} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "gaussian", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: kaneyoshi

Function used in Kaneyoshi's papers on amorphous magnets.

.. math::
    J(r) = J_0 (r-r_0)^2\operatorname{exp}\left( -\frac{(r-r_0)^2}{2\sigma^2} \right)

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma
        ("A", "A", "kaneyoshi", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: rkky

RKKY type interaction with oscillating sign. Defined with a shift :math:`r_0` and
wavenumber :math:`k`.

.. math::
    J(r) = -J_0 \frac{2 k (r-r_0) \cos(2 k (r-r_0)) - \sin(2 k (r-r_0)}{(2 k (r - r_0))^4}

.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, wavenumber
        ("A", "A", "rkky", 3.0, 20.0, 1.0, 1.0)
    );

.. describe:: gaussian_multi

Multiple (three) gaussian functions with independent centers, widths and amplitudes.

.. math::
    J(r) = J_0 \operatorname{exp}\left( -\frac{(r-r_0)^2}{2\sigma_0^2} \right) + J_1 \operatorname{exp}\left( -\frac{(r-r_1)^2}{2\sigma_1^2} \right) + J_2 \operatorname{exp}\left( -\frac{(r-r_2)^2}{2\sigma_2^2} \right)


.. code-block:: none

    interactions = (
        // type_i, type_j, functional, r_cutoff, J0, r0, sigma0, J1, r1, sigma1, J2, r2, sigma2
        ("A", "A", "gaussian_multi", 4.0, 20.0, 1.0, 1.0, -10.0, 2.0, 1.0, 5.0, 3.0, 1.0)
    );

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