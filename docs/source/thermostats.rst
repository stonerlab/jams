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
at fixed scalar temperature. Classical white noise, flat FFT spectra,
``quantum-ohmic`` and other SPDE configurations use the solver's existing
scheme.

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
