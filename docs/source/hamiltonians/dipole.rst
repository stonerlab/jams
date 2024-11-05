dipole
======

Magnetic dipole-dipole interactions

.. math::
      \mathscr{H} = -\tfrac{1}{2}\sum_{\substack{i \neq j \\ |r_{ij}| \leq r_{\rm cutoff}}} \frac{\mu_{s,i}\mu_{s,j}\mu_0}{4\pi} \frac{3(\mathbf{S}_{i}\cdot\hat{\mathbf{r}}_{ij})(\mathbf{S}_{j}\cdot\hat{\mathbf{r}}_{ij}) - \mathbf{S}_{i}\cdot\mathbf{S}_{j}}{|r_{ij}|^3}

where
    - :math:`r_{\rm cutoff}` is a spherical cutoff radius for the interaction
    - :math:`\mu_{s,i}` is the magnetic moment of spin :math:`i` in Bohr magnetons
    - :math:`\mu_0 = 4\pi\times 10^{-7} ~ {\rm H/m}` is the vacuum permeability
    - :math:`\mathbf{S}_{i}` is a classical spin vector of unit length
    - :math:`\mathbf{r}_{ij} = \mathbf{r}_{j} - \mathbf{r}_{i}` is the vector from site :math:`i` to site :math:`j` in meters
    - :math:`\hat{\mathbf{r}}_{ij}` is unit vector in the direction of :math:`\mathbf{r}_{ij}`

The factor of :math:`\tfrac{1}{2}` accounts for double counting.

Variants
########

There are multiple dipole Hamiltonians which perform the same calculation but xusing different methods which may be
optimal for different solver types, for example Monte Carlo solvers need to calculate energy differences one at a time,
but a dynamical solver can have all fields calculated simultaneously.

.. describe:: dipole-bruteforce

    A naive evaluation of the double sum. Lowest memory usage but very slow. Useful mostly for testing correctness of
    more complex methods.


.. describe:: dipole-neartree

    A semi-bruteforce method which uses a fast dynamic neighbour lookup (the near tree). It has low memory usage but is
    slower than the more memory hungry methods. Useful for Monte Carlo when limited by memory.

.. describe:: dipole-neighbour-list

    A middle point between the dipole-neartree and the dipole-tensor. It stores the full neighbour list and
    :math:`\hat{\mathbf{r}}_{ij}` but calculates each the dipole expression each time. Uses about 1/3 of the memory
    of the dipole-tensor and is much faster than dipole-neartree. Often an optimal choice for Monte-Carlo if enough
    memory is available.

.. describe:: dipole-tensor

    Stores the full dipole interaction tensor (i.e. with all the dipole terms precomputed). Extremely expensive in
    memory but very fast. Only really useful for small systems or short :math:`r_{\rm cutoff}`.

.. describe:: dipole-fft

    Calculates the dipole tensor in Fourier space and applies it using convolution theorem. Very efficient in memory
    and very fast, but calculates all spins simultaneously. This overhead makes it too slow for Monte-Carlo, but very
    efficient for dynamical solvers.

Settings
########

.. describe:: r_cutoff

    Spherical cutoff radius for dipole interactions in units of lattice parameters