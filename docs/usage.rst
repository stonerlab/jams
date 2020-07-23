Usage
=====

JAMS is controlled by inputting configuration files or strings in the
`libconfig format <http://hyperrealm.github.io/libconfig/libconfig_manual.html#Configuration-Files>`_ .
For example:

.. code-block:: none

  ./jams input.cfg

Configuration
-------------

The configuration files and strings are read in the order they appear on the command line. Any new settings which
appear are added to the configuration. Any duplicated settings are overwritten in the configuration. For example:

.. code-block:: none

  ./jams input.cfg 'physics : {temperature = 100.0;};'

overrides the :code:`physics.temperature` setting of :code:`input.cfg`. This provides a simple way to write batch
scripts to loop over parameters or chain together multiple simulations with different steps. Multiple configuration
files can be chained together so that simulations can be composed, for example the definition of the unit cell might
be contained in one file and the choice of solve may be in another file.

A combined configuration file will be written with the suffix :code:`_combined.cfg`, which contains the final
configuration after all configuration files and strings have been merged and represents the simulation configuration
which was actually used.

.. note::
    libconfig files can be created and read with Python using `libconf <https://pypi.org/project/libconf/>`_
    which allows easy access to the configuration settings in post process analysis.

The final configuration must include the following settings (in any order):

- :ref:`solver <solvers>`
- :ref:`unitcell <unitcell>`
- :ref:`lattice <lattice>`
- :ref:`materials <materials>`
- :ref:`hamiltonians <hamiltonians>`
- :ref:`monitors <monitors>`
- physics

The details of the syntax and further requirements for each one is given elsewhere in the documentation.

Command line flags
------------------

.. describe:: --setup-only

    Runs JAMS but quits before the solving starts. i.e. performs the full system initialisation only. This is useful for
    testing/debugging or extracting information about a system without solving.

.. describe:: --output=<path>

    Sets the path to output data files to. If the path does not exist JAMS will attempt to make all folders in the path.
    This option is useful for jobscripts and workflow tools to avoid changing directories. Input will still be read
    relative to the current working directory.

.. describe:: --name=<simulation_name>

    Sets the simulation name which is prefixed to output files written by JAMS. If this is not set the config file name
    is used.

Output
------

Output files will be written either to the directory where JAMS is run or the location given by :code:`--output`.
The files will be prefixed by the simulation name which will be (in order of precidence): The name given by
:code:`--name`, the name of the first configuration file,  the name `jams` (if the configuration is fully specified
with strings).

General system setup information is written to the terminal (`cout`). This should be redirected to a file if you want to
save it. This is useful for example to check what symmetry has been found or to check the number of exchange
interactions found is what you expect.