Usage
=====

JAMS is controlled by an input configuration file. Running jams involves

.. code-block:: none

  ./jams input.cfg

Settings in the configuration file can also be overwritten or added by include a patch string at the end of the command
line arguments. For example:

.. code-block:: none

  ./jams input.cfg 'physics : {temperature = 100.0;};'

This provides a simple way to write batch scripts to loop over parameters or chain together multiple simulations with
different steps.
