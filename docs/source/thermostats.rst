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
per-spin profiles because its filter is built for one fixed scalar temperature.
