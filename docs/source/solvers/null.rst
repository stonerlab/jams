null
====

The ``null`` solver performs no physical time integration. It exists so that
the rest of the simulation setup can run, including Hamiltonian construction
and monitor output, without evolving the spin system.

This is useful for:

- setup-only style runs from configuration files,
- writing monitor output for the initial state,
- debugging Hamiltonians and initial conditions.

Configuration
-------------

.. code-block:: cfg

   solver = {
     module = "null";
   };

The solver runs exactly one iteration and leaves the spin configuration
unchanged.
