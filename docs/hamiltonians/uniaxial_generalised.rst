uniaxial-generalised
========

The base uniaxial Hamiltonian does not allow rotational symmetry along the uniaxial anisotropy.
But, these terms are allowed--by symmetry--in atomic lattices which are subgroups of :math:`6/mmm`
(hexagonal). To fourth order in direction cosines, which we label :math:`\vec{u}`, :math:`\vec{v}`
, and :math:`\vec{w}`, these are

.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Anisotropy
     - Rank
     - Rotational symmetry along :math:`\vec{w}`
     - Number
     - Maximal crystal class
   * - :math:`(\vec{S} \cdot \vec{w})^2`
     - 2
     - 0
     - 0
     - Hexagonal
   * - :math:`(\vec{S} \cdot \vec{w})^4`
     - 4
     - 0
     - 0
     - Hexagonal
   * - :math:`(\vec{S} \cdot \vec{w})(\vec{S} \cdot \vec{v})[3(\vec{S} \cdot \vec{u})^2 - (\vec{S} \cdot \vec{v})^2]`
     - 4
     - 3
     - 0
     - Trigonal
   * - :math:`(\vec{S} \cdot \vec{u})^4+(\vec{S} \cdot \vec{v})^4`
     - 4
     - 4
     - 0
     - Tetragonal
   * - :math:`(\vec{S} \cdot \vec{u})^2 (\vec{S} \cdot \vec{v})^2`
     - 4
     - 4
     - 1
     - Tetragonal
   * - :math:`(\vec{S} \cdot \vec{u})^4- 6(\vec{S} \cdot \vec{u})^2(\vec{S} \cdot \vec{v})^2 + (\vec{S} \cdot \vec{v})^4`
     - 4
     - 4
     - 2
     - Tetragonal

These are irreducible, they all have different shapes and one cannot be created as a sum of the others. Additional terms
such as specific forms of easy-cones are solutions to the symmetry problem but these are not irreducible.

For crystals with orthorhombic, monoclinic, or triclinic point groups there are more allowed terms. But these are sums
of the above terms with different values of :math:`\vec{u}`, :math:`\vec{v}` , and :math:`\vec{w}`.


Settings
########

Below is an example input in a config file.

.. code-block:: none

{
    module = "uniaxial-generalised"
    identifier = (2,0,0)
    anisotropies = (
        (1, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1e-24),
        (2, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1e-24),
    );
    unit_name = "joules";
}

.. describe:: identifier

    The identifier variable has no default value and is required. A list in parentheses of three integers is expected. Following the
    table above, from left to right the three integers represent:
    - The rank/order of the anisotropy.
    - The number of directions within the :math:`uv`-plane which have the same energy, 0 returns an anisotropy which is
      isotropic in the plane.
    - Anisotropy number. This is only non-zero if there are multiple anisotropy terms of the same rank and rotational
      symmetry. Refer to the table above.

.. describe:: anisotropies

    This follows the convention of the cubic anisotropy Hamiltonian. :code:`anisotropies` is a list in parentheses of
    anisotropy elements. The :code:`anisotropies` list must be the same length as the number of materials specified.
    Anisotropy elements are a list, their elements from left to right are:
    - Material number / material name. 1 is the first material in the config. Alternatively this can be named, "Fe" for
      example.
    - :math:`\vec{u}`. This is a vector with three elements which corresponds to one of the in-plane vectors.
    - :math:`\vec{v}`. The other in-plane vector.
    - :math:`\vec{w}`. The out-of-plane vector.
    - Anisotropy constant :math:`K`. The default units are Joules.

    The above are listed per element. Symmetry of the crystal determines which anisotropies are allowed.
    Local environment of an element determines vectors and strengths.

.. describe:: unit_name

    This is an optional parameter which allows the units of the anisotropy constant to be specified. Default value is
    joules.

Fields
########

Below are the derivatives of the energy terms above

.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Anisotropy
     - Field
   * - :math:`(\vec{S} \cdot \vec{w})^2`
     - :math:`2\vec{w}(\vec{S} \cdot \vec{w})`
   * - :math:`(\vec{S} \cdot \vec{w})^4`
     - :math:`4\vec{w}(\vec{S} \cdot \vec{w})^3`
   * - :math:`(\vec{S} \cdot \vec{w})(\vec{S} \cdot \vec{v})[3(\vec{S} \cdot \vec{u})^2 - (\vec{S} \cdot \vec{v})^2]`
     - :math:`\vec{u}[6(\vec{S} \cdot \vec{u})(\vec{S} \cdot \vec{v})(\vec{S} \cdot \vec{w})] + \vec{v}[(\vec{S} \cdot \vec{v})(3(\vec{S} \cdot \vec{u})^2 - (\vec{S} \cdot \vec{v})^2 - 2(\vec{S} \cdot \vec{w})(\vec{S} \cdot \vec{v}))] + \vec{w}[(\vec{S} \cdot \vec{v})(3(\vec{S} \cdot \vec{u})^2-(\vec{S} \cdot \vec{v})^2)]`
   * - :math:`(\vec{S} \cdot \vec{u})^4+(\vec{S} \cdot \vec{v})^4`
     - :math:`\vec{u}[4(\vec{S} \cdot \vec{u})^3] + \vec{v}[4(\vec{S} \cdot \vec{v})^2]`
   * - :math:`(\vec{S} \cdot \vec{u})^2 (\vec{S} \cdot \vec{v})^2`
     - :math:`\vec{u}[2(\vec{S} \cdot \vec{u})(\vec{S} \cdot \vec{v})^2] + \vec{v}[2(\vec{S} \cdot \vec{u})^2(\vec{S} \cdot \vec{v})]`
   * - :math:`(\vec{S} \cdot \vec{u})^4- 6(\vec{S} \cdot \vec{u})^2(\vec{S} \cdot \vec{v})^2 + (\vec{S} \cdot \vec{v})^4`
     - :math:`\vec{u}[(\vec{S} \cdot \vec{u})(4(\vec{S} \cdot \vec{u})^2-6(\vec{S} \cdot \vec{v})^2)] + \vec{v}[(\vec{S} \cdot \vec{v})(4(\vec{S} \cdot \vec{v})^2-6(\vec{S} \cdot \vec{u})^2)]`