llg-heun-gpu
==========================

Solves the Landau-Lifshitz-Gilbert (LLG) equation using the Heun integration
method on a GPU.

.. note::
    JAMS must be build with CUDA supprot and the computer must have an NVIDIA
    GPU supporting CUDA available.

Required settings
^^^^^^^^^^^^^^^^^

.. option:: t_step

Integration time step in seconds. Usually :math:`1\times10^{-16}\mathrm{s}` is sufficient.

.. option:: t_max

Maximum integration time in seconds. It can be stopped early by a converged
monitor.

Optional settings
^^^^^^^^^^^^^^^^^

.. option:: t_min

Minimum integration time in seconds. Solver will continue to this time even
if a monitor has converged.