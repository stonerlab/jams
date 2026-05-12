.. _torque-monitor:

torque
======

The torque monitor writes ``torq.tsv`` with the average torque per spin from
each Hamiltonian term.

``grouping`` controls which spins are averaged together:

``none`` or ``total``
  Output one total-system torque for each Hamiltonian term. This is the
  default and preserves the historical column names.

``materials``
  Output one torque per material for each Hamiltonian term.

``positions``
  Output one torque per unit cell basis position for each Hamiltonian term.
