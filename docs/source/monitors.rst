.. _monitors:

Monitors
========

Time coordinates are in picoseconds. TSV time columns use 17 significant
digits so that reading them back preserves the double-precision clock value.
The monitor's ``precision`` setting continues to control observable columns
independently of time. This can make time columns wider than other columns;
read TSV files using tabs or whitespace rather than fixed character positions.

XDMF time values and the VTU time annotation also preserve double precision.
CUDA spin-current XDMF time is in picoseconds, consistent with its HDF5 time
attribute (older versions incorrectly multiplied this XDMF value by ``1e12``).
Unit-cell and spin-current snapshots store the actual time and iteration in
their HDF5 attributes; older versions created these attributes without writing
their values.
See :ref:`solvers` for duration rounding and the regular sampling convention.

.. toctree::
   :glob:
   :maxdepth: 2

   monitors/*
