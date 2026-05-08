anisotropy-polynomial
=====================

The polynomial anisotropy Hamiltonian describes single-ion anisotropy as an
expansion in real tesseral harmonic polynomials. It is useful for uniaxial,
transverse, cubic-like, and lower-symmetry anisotropies that can be written as
polynomials of the spin direction.

For spin :math:`i`, JAMS evaluates

.. math::

    \mathcal{H}_i =
        \sum_{l=2,4,6} \sum_{m=-l}^{l}
        C_{l,m}^{(i)}
        Z_{l,m}
        \left(
            \mathbf{s}_i \cdot \mathbf{u}_i,
            \mathbf{s}_i \cdot \mathbf{v}_i,
            \mathbf{s}_i \cdot \mathbf{w}_i
        \right),

where :math:`\mathbf{s}_i` is the spin direction, normally a unit vector, and
:math:`\mathbf{u}_i`, :math:`\mathbf{v}_i`, and :math:`\mathbf{w}_i` are the
local reference axes for that spin. The implemented terms are
:math:`l=2,4,6` with :math:`-l \le m \le l`.

The Hamiltonian module name is :code:`"anisotropy-polynomial"`.


Tesseral Polynomial Basis
#########################

The runtime basis is the monic tesseral polynomial basis. In this basis the
leading polynomial coefficient is chosen to be simple, and other supported
normalisation conventions are converted to this basis when the Hamiltonian is
constructed.

If the input uses normalisation convention :math:`N`, JAMS uses a scale factor
:math:`a_{l,m}^{N}` defined by

.. math::

    Z_{l,m}^{N}(x,y,z) = a_{l,m}^{N} Z_{l,m}^{\mathrm{monic}}(x,y,z),

and stores the coefficient

.. math::

    C_{l,m}^{\mathrm{monic}} = a_{l,m}^{N} C_{l,m}^{N}.

The runtime energy is then always evaluated as
:math:`C_{l,m}^{\mathrm{monic}} Z_{l,m}^{\mathrm{monic}}`.

For example, the axial :math:`m=0` monic polynomials are

.. math::

    Z_{2,0}(z) &= z^2 - \frac{1}{3}, \\
    Z_{4,0}(z) &= z^4 - \frac{6}{7}z^2 + \frac{3}{35}, \\
    Z_{6,0}(z) &= z^6 - \frac{15}{11}z^4 + \frac{5}{11}z^2 - \frac{5}{231}.

The :math:`l=2` non-axial monic terms are

.. math::

    Z_{2,-2}(x,y,z) &= xy, \\
    Z_{2,-1}(x,y,z) &= yz, \\
    Z_{2,1}(x,y,z) &= xz, \\
    Z_{2,2}(x,y,z) &= x^2 - y^2.

Higher-order :math:`l=4` and :math:`l=6` terms are evaluated by the same
tesseral polynomial implementation used by the CPU and CUDA Hamiltonians.

The monic basis is the internal basis because it gives the evaluator one
canonical set of polynomial functions. Input coefficients in Racah, Stevens or
Condon-Shortley conventions are rescaled once at setup time, so the CPU and
CUDA kernels do not need separate implementations for each convention. This is
also the basis used by the crystal-field Hamiltonian after it converts its
angular functions into tesseral form.


Fields
######

The field returned by this Hamiltonian is the unconstrained Cartesian energy
gradient field

.. math::

    \mathbf{H}_i = -\frac{\partial \mathcal{H}_i}{\partial \mathbf{s}_i}.

It is not projected onto the tangent plane of the unit sphere. This is
intentional: each Hamiltonian contributes an unconstrained derivative, and spin
length constraints should be applied consistently by the solver or optimiser.
Many spin dynamics solvers impose the constraint through cross products,
renormalisation, or a solver-level projection. If an algorithm needs a tangent
descent direction on :math:`|\mathbf{s}|=1`, apply the projection once to the
total field from all Hamiltonian terms, not separately inside individual
Hamiltonians.


Settings
########

.. describe:: energy_units (optional | string)

    Energy units of the coefficients. If omitted, JAMS uses the default energy
    unit.

.. describe:: normalisation (optional | string)

    Normalisation convention used by the input coefficients. The American
    spelling :code:`normalization` is also accepted. The default is
    :code:`"monic"`.

    Supported values are:

    :code:`"monic"`
        Coefficients multiply the internal monic tesseral polynomials directly.
        For example, the :math:`l=2,m=0` term is :math:`z^2 - 1/3`.

    :code:`"condon-shortley"`
        Coefficients multiply unit-normalised real tesseral harmonics using the
        Condon-Shortley phase convention. JAMS converts these coefficients to
        the internal monic basis.

    :code:`"racah"`
        Coefficients multiply Racah-normalised real tesseral harmonics,
        where :math:`\mathrm{CS}` denotes the unit-normalised
        Condon-Shortley real tesseral harmonics:

        .. math::

            Z_{l,m}^{\mathrm{Racah}} =
                \sqrt{\frac{4\pi}{2l+1}} Z_{l,m}^{\mathrm{CS}},

        For example, the :math:`l=2,m=0` term is
        :math:`(3z^2 - 1)/2`. The aliases :code:`"wybourne"`,
        :code:`"racah-wybourne"`, :code:`"wybourne-racah"` and
        :code:`"crystal-field"` are also accepted.

    :code:`"stevens"`
        Coefficients multiply the classical Stevens tesseral polynomial
        convention. For example, the :math:`l=2,m=0` term is
        :math:`3z^2 - 1`. The alias :code:`"stevens-operators"` is also
        accepted.

.. describe:: anisotropies (required | list)

    List of anisotropy definitions. Each definition applies to either a
    material name or a unit-cell basis index.

    The general form is

    .. code-block:: none

        (target, u, v, w, coefficient...)

    where :code:`target` is a material name such as :code:`"Fe"` or a
    one-based unit-cell basis index such as :code:`1`. The local axes
    :code:`u`, :code:`v` and :code:`w` are three-component arrays.

    Each coefficient has the form

    .. code-block:: none

        (l, m, C_lm)

    where :code:`l` is :code:`2`, :code:`4` or :code:`6`, and
    :code:`-l <= m <= l`.

    Axes can be omitted. If no axes are given, the default local frame is

    .. code-block:: none

        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        w = [0.0, 0.0, 1.0]

    For purely axial anisotropy, where all non-zero terms that apply to a spin
    have :math:`m=0`, a single :code:`w` axis can be supplied:

    .. code-block:: none

        (target, w, coefficient...)

    This short form is invalid if any non-zero term for the same spin has
    :math:`m \ne 0`, because non-axial terms require a full local frame.


Axis Rules
##########

All supplied axes are normalised on input. If full :code:`u`, :code:`v` and
:code:`w` axes are supplied, they must be mutually orthogonal.

Multiple anisotropy definitions may apply to the same spin. This is allowed,
and coefficients with the same :math:`l,m` are added together. However, the
explicit axes that apply to a given spin must be consistent. It is malformed
input to define multiple anisotropies for the same spin with different local
frames. For one-axis axial definitions, consistency is checked using the
:code:`w` axis.

If one definition for a spin omits axes and another matching definition gives
explicit axes, the explicit axes are used for the combined anisotropy of that
spin.


Energy Offsets
##############

The monic tesseral polynomials include constant terms. For example
:math:`Z_{2,0}=z^2-1/3`. These constants affect the absolute energy reported by
the Hamiltonian, but they do not affect the field or energy differences between
spin directions for a fixed set of coefficients.


Examples
########

Axial second-order anisotropy
-----------------------------

This example applies an axial :math:`l=2,m=0` anisotropy to material
:code:`"A"` along the global :math:`z` axis.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        anisotropies = (
          ("A", [0.0, 0.0, 1.0], (2, 0, 1.0))
        );
      }
    );

Since only :math:`m=0` is used, only the axial :code:`w` axis is needed.


Axial fourth- and sixth-order terms
-----------------------------------

Higher-order axial anisotropy can be written by adding :math:`l=4,m=0` and
:math:`l=6,m=0` terms.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        normalisation = "stevens";
        anisotropies = (
          ("Nd", [0.0, 0.0, 1.0],
            (2, 0, 1.25),
            (4, 0, -0.08),
            (6, 0, 0.003))
        );
      }
    );

Here the input coefficients use Stevens normalisation. JAMS converts them to
the internal monic basis before evaluation.


Full local frame with non-axial terms
-------------------------------------

Non-axial terms need a complete local frame. The following example rotates the
local axes so that local :math:`w` is the global :math:`x` direction.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        anisotropies = (
          ("A",
            [0.0, 1.0, 0.0],  // u
            [0.0, 0.0, 1.0],  // v
            [1.0, 0.0, 0.0],  // w
            (2, 0, 1.0),
            (2, 2, 0.2),
            (4, -2, -0.05))
        );
      }
    );


Different materials
-------------------

Different materials can have different anisotropies. Definitions are matched
using the material names from the lattice configuration.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        normalisation = "racah";
        anisotropies = (
          ("A", [0.0, 0.0, 1.0], (2, 0, 0.40)),
          ("B", [1.0, 0.0, 0.0], (2, 0, -0.15), (4, 0, 0.02))
        );
      }
    );


Unit-cell basis indices
-----------------------

The target can also be a one-based unit-cell basis index. This is useful when
the same material appears at inequivalent basis positions.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        anisotropies = (
          (1, [0.0, 0.0, 1.0], (2, 0, 0.5)),
          (2, [1.0, 0.0, 0.0], (2, 0, 0.3))
        );
      }
    );


Splitting definitions
---------------------

Definitions that apply to the same spin are accumulated. This can be useful
when separating contributions by source.

.. code-block:: none

    hamiltonians = (
      {
        module = "anisotropy-polynomial";
        energy_units = "meV";
        anisotropies = (
          ("A", [0.0, 0.0, 1.0], (2, 0, 1.0)),
          ("A", (4, 0, 0.1)),
          ("A", (6, 0, -0.002))
        );
      }
    );

The omitted axes in the second and third definitions inherit the explicit
axis from the first definition for spins of material :code:`"A"`.


Malformed input examples
########################

The following is invalid because two axes are supplied. The input must contain
either one axial :code:`w` axis, all three axes, or no axes.

.. code-block:: none

    anisotropies = (
      ("A", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], (2, 0, 1.0))
    );

The following is invalid because a single axis is supplied but the term
:math:`m=2` is non-axial.

.. code-block:: none

    anisotropies = (
      ("A", [0.0, 0.0, 1.0], (2, 2, 1.0))
    );

The following is invalid because different explicit axes apply to the same
spin.

.. code-block:: none

    anisotropies = (
      ("A", [0.0, 0.0, 1.0], (2, 0, 1.0)),
      ("A", [1.0, 0.0, 0.0], (4, 0, 0.1))
    );


Implementation Notes
####################

The CPU and CUDA implementations use the same tesseral polynomial evaluator.
At construction time, JAMS builds unique anisotropy profiles. Each spin stores
only a profile index. A profile contains:

* local axis data,
* the combined axial polynomial coefficients,
* a sparse list of residual non-axial tesseral keys and coefficients.

The common axial :math:`m=0` terms are evaluated as

.. math::

    E(z) = A_0 + A_2 z^2 + A_4 z^4 + A_6 z^6,

with

.. math::

    H_z = -\frac{dE}{dz}.

This avoids a tesseral key lookup for the most common case. CUDA kernels also
launch over the list of active spins whose profile has non-zero terms. Inactive
spins contribute zero energy and zero field.
