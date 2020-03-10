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

Additional arguments
---------
.. describe:: --setup-only

    Runs JAMS but quits before the solving starts. i.e. performs the full system initialisation only. This is useful for
    testing/debugging or extracting information about a system without solving.

.. describe:: --output=<path>

    Sets the path to output data files to. If the path does not exist JAMS will attempt to make all folders in the path.
    This option is useful for jobscripts and workflow tools to avoid changing directories. Input will still be read
    relative to the current working directory.
