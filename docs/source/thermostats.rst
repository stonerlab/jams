Thermostats
===========

Stage sampling
--------------

``solver.thermostat_sampling`` defaults to ``"auto"``. The following solvers
sample the same coloured field realisation at their RHS evaluation times.
Times are offsets within a solver step of length :math:`h`.

.. list-table::
   :header-rows: 1

   * - Solver
     - Stage times
     - Bath interval
   * - CPU and GPU Heun
     - :math:`0, h`
     - :math:`h`
   * - CPU and GPU RKMK2, GPU semi-implicit
     - :math:`0, h/2`
     - :math:`h/2`
   * - GPU DM
     - :math:`0, h`
     - :math:`h/2`
   * - CPU and GPU RKMK4, GPU LLG, LLG-SOT and GSE RK4
     - :math:`0, h/2, h/2, h`
     - :math:`h/2`

Repeated stages share a sample. The bath advances across intervening ticks
and carries the endpoint field into the next step.

Eligible baths are quantum SPDE with zero-point noise disabled and a fixed
scalar or per-spin temperature, and the GPU Lorentzian and thermal-only
quantum Ohmic FFT spectra (``quantum-no-zero-ohmic`` or ``bose-einstein``)
at fixed scalar temperature. Solvers handle white fields and other spectra
with their step-based schemes.

Set ``solver.thermostat_sampling = "legacy";`` to recover the previous
scheme and bath cadence for a given solver and bath. CUDA RS2 uses its
existing stochastic tableau for white noise and its existing stage
sampling for coloured fields. Retarded Lorentzian dynamics uses its
existing scheme.

The timestep must resolve the coloured bath's significant frequencies.
LLG, LLG-SOT and GSE RK4 refine eligible FFT baths from :math:`h` to
:math:`h/2` while retaining the legacy spectral cutoff at
:math:`1/(2h)`. The accuracy of stage integration depends on the regularity
and resolution of the supplied field.

RKMK white-noise half-steps
---------------------------

With ``solver.thermostat_sampling = "auto"``, CPU and GPU RKMK2 and RKMK4
use a variance-corrected Cayley update for classical white thermostats
and for flat FFT spectra (``classical`` and ``classical-ohmic``). One
independent field draw is used for each thermal half-step on either side
of the deterministic RKMK step. The thermostat cadence and Hamiltonian
evaluation count are unchanged. ``"legacy"`` selects the original
Rodrigues kicks. Baths that supply correlated fields follow the stage
sampling or step-based scheme described above.

For a spin of moment :math:`\mu` at temperature :math:`T`, the noise-only
LLG equation has generator :math:`D\Delta_{S^2}` with
:math:`D=\alpha\gamma_\mathrm{eff}k_\mathrm{B}T/\mu`.
Use the site's effective gyromagnetic ratio, moment and bath temperature
in this expression. Both kicks use the bath temperature for that
solver step, including per-spin temperature profiles where supported.

For a half-step :math:`\tau=h/2`, set :math:`x=D\tau`. Calculate the
tangent increment :math:`\mathbf u=\tau\boldsymbol{\omega}_\mathrm{noise}
                       \times\mathbf s` from the existing Gaussian field draw,
where :math:`\boldsymbol{\omega}_\mathrm{noise}
 =\gamma_\mathrm{eff}(\mathbf H_\mathrm{noise}
       +\alpha\mathbf s\times\mathbf H_\mathrm{noise})`.
Its two tangent components each have variance :math:`2x`. The update is

.. math::

   \mathbf v=\sqrt{1+x}\,\mathbf u,\qquad
   a=\frac{\|\mathbf v\|^2}{4},\qquad
   \mathbf s^+=\frac{(1-a)\mathbf s+\mathbf v}{1+a}.

The implementation preserves the input spin length in exact arithmetic.
For nonunit input spins it uses :math:`a=\|\mathbf v\|^2/(4\|\mathbf s\|^2)`,
and a vacancy stays at zero. The following moment estimates apply to
unit spins. The factor
:math:`\sqrt{1+x}` adjusts the tangent variance to cancel the Cayley
map's leading bias in the mean angular deflection.
For :math:`c=\mathbf s\cdot\mathbf s^+` it matches
:math:`\mathbb E[1-c]=2x-2x^2+O(x^3)` and
:math:`\mathbb E[(1-c)^2]=8x^2+O(x^3)` for spherical diffusion.
Rotational symmetry gives local weak error :math:`O(\tau^3)` for smooth
observables. At fixed bath parameters, the Strang composition is weak
second order with either RKMK core. The correction is derived for
:math:`D\tau\ll1` and targets ensemble statistics.

Position-dependent temperature
------------------------------

LLG thermostats support a scalar global temperature through ``physics.temperature``
as before.  A position-dependent temperature can be configured by adding
``thermostat.temperature_regions``.  Region coordinates use the same Cartesian
internal lattice coordinates as the spin positions.  Region boxes are half-open:
``origin <= position < origin + size``.

Every spin must be covered by exactly one region unless
``thermostat.default_temperature`` is set.  Regions may be either constant
temperature regions or linear gradients:

.. code-block:: cfg

  thermostat = {
    temperature_regions = (
      {
        type = "constant";
        origin = [0.0, 0.0, 0.0];
        size = [10.0, 20.0, 5.0];
        temperature = 300.0;
      },
      {
        type = "linear";
        origin = [10.0, 0.0, 0.0];
        size = [10.0, 20.0, 5.0];
        direction = [1.0, 0.0, 0.0];
        low = 300.0;
        high = 500.0;
      }
    );
  };

For linear regions, ``low`` is applied at the lower projection of the region
box along ``direction`` and ``high`` at the upper projection.

The scalar-temperature fast path is retained when ``temperature_regions`` is not
set.  Per-spin temperature profiles are currently supported by the classical
CPU/GPU LLG thermostats and ``quantum-spde-gpu``.  ``general-fft-gpu`` rejects
per-spin temperature profiles for coloured spectra because its filter is
built for one fixed scalar temperature.

Gyromagnetic ratio and damping
------------------------------

The material gyromagnetic ratio :math:`\gamma` is stored in rad ps\ :sup:`-1`
T\ :sup:`-1` and :math:`\alpha` is dimensionless. The solver setting
``gilbert_prefactor`` selects :math:`\gamma_\mathrm{eff}=\gamma/(1+\alpha^2)`
when enabled and :math:`\gamma_\mathrm{eff}=\gamma` otherwise.

For unit-spin LLG the white field spectrum is :math:`2 k_\mathrm{B}T\eta`,
with :math:`\eta=\alpha/[\mu\gamma_\mathrm{eff}(1+\alpha^2)]`. The
variable-length GSE uses additive noise and
:math:`\eta=\alpha/(\mu\gamma_\mathrm{eff})`. Ohmic coloured baths use
the corresponding coupling to scale the target spectrum.

``ll-lorentzian-rk4-gpu`` defaults to a classical Lorentzian FFT bath.
The spectrum and retarded damping kernel share the coupling
:math:`\eta=\alpha/(\mu\gamma)` and the configured resonance and linewidth.
The flag changes the precession rate. The resulting low-frequency
dimensionless damping is :math:`\mu\gamma_\mathrm{eff}\eta`. An explicitly
selected unmatched thermostat supplies a prescribed driving field.
