Thermostats
===========

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
