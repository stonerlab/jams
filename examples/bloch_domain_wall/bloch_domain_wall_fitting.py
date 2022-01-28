#!/usr/bin/env python3

# 2022-01-27 Joseph Barker (j.barker@leeds.ac.uk)
# This python script is for fitting Bloch domain walls in JAMS h5 output files.
# For a given simulation name it read each spin output h5 file and fit the mx,my,mz
# data with a Bloch wall profile, where it is assumed the Bloch wall is unconstrained,
# and the magnetisation at the left/right sides are in the -/+z directions.
#
# NOTE: We are using the convention for width which includes a factor of
# pi. i.e. = pi \sqrt{A/K}. This factor of pi is an arbitrary choice in the
# definition of width. Some authors use it some don't. Including the pi gives
# are much better feeling for the extent of the domain wall when trying to fit
# it into a finite size simulation box.
#
# Requirements
# ------------
# - numpy
# - h5py
# - lmfit
#
# Usage
# -----
# usage: bloch_domain_wall_fitting.py [-h] [-o OUTFILE] [-p] simulation_name
#
# positional arguments:
#   simulation_name       input JAMS simulation name

# optional arguments:
#   -h, --help            show this help message and exit
#   -o OUTFILE, --output OUTFILE
#                         output file of domain wall properties
#   -p, --print           output pdf of the fitting

import os
import h5py
import lmfit
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def bloch_domain_wall_mx(x, center, height, width):
    """
    Bloch domain wall x-magnetisation profile

    x:       x coordinate
    center: domain wall center
    height: domain wall height (magnetization along z)
    width:  domain wall width
    """
    return 0.0*x


def bloch_domain_wall_my(x, center, height, width):
    """
    Bloch domain wall y-magnetisation profile

    x:       x coordinate
    center: domain wall center
    height: domain wall height (magnetization along z)
    width:  domain wall width
    """
    return height/np.cosh(np.pi * (x-center)/width)


def bloch_domain_wall_mz(x, center, height, width):
    """
    Bloch domain wall z-magnetisation profile

    x:       x coordinate
    center: domain wall center
    height: domain wall height (magnetization along z)
    width:  domain wall width
    """
    return height*np.tanh(np.pi * (x-center)/width)


def bloch_domain_wall_residual(params, x, mx, my, mz):
    """
    Residual function for use with lmfit.minimize to fit a Bloch domain wall
    using mx, my and mz data. 

    Although the bloch_domain_wall_mx is zero, but including it in the residual we get a better
    estimate of the errors because they include information about how far the
    x profile has deviated from a flat line.
    """
    model_x = bloch_domain_wall_mx(x, params['center'], params['height'], params['width'])
    model_y = bloch_domain_wall_my(x, params['center'], params['height'], params['width'])
    model_z = bloch_domain_wall_mz(x, params['center'], params['height'], params['width'])

    resid_x = mx - model_x
    resid_y = my - model_y
    resid_z = mz - model_z

    return np.concatenate((resid_x, resid_y, resid_z))



parser = ArgumentParser()

parser.add_argument("simulation_name", nargs=1,
                    help="input JAMS simulation name")

parser.add_argument("-o", "--output", dest="outfile", required=False,
                    help="output file of domain wall properties")

parser.add_argument("-p", "--print", dest="print_fits", action='store_true',
                    help="output pdf of the fitting")

args = parser.parse_args()

# Remove any extension incase we've given the .cfg file rather than just
# the simulation name
simulation_name = os.path.splitext(args.simulation_name[0])[0]


# The output file will be based on the simulation name unless specified
# in the arguments
if args.outfile:
    output_filename = args.outfile
else:
    output_filename = f'{simulation_name}_domain_wall.tsv'


lattice_filename = f"{simulation_name}_lattice.h5"
spin_filename = f"{simulation_name}_0000100.h5"

with h5py.File(lattice_filename, "r") as f:
    positions = np.array(f['positions'])

xplanes = np.unique(positions[:,0])

fit_params = lmfit.Parameters()
fit_params.add('center', value = np.mean(xplanes))
fit_params.add('height', value = 256.0)
fit_params.add('width',  value = 40.0)



with open(output_filename, 'w') as outfile:
    print('output dw_center_nm dw_center_nm_stderr dw_mz_muB dw_mz_muB_stderr dw_width_nm dw_width_nm_stderr', file=outfile)
    for n in range(0,9999999):
        # rather than working out how many output files exist in the series in advance
        # we simply try the maximum number of times and then break when we can't open
        # a file
        try:
            filename = f'{simulation_name}_{n:07}.h5'

            with h5py.File(filename, 'r') as f:
              spins = np.array(f['spins'])

            # sum the magnetisation in each xplane
            magnetisation = dict([(k, 0.0) for k in xplanes])

            for r, s in zip(positions, spins):
                magnetisation[r[0]] = magnetisation[r[0]] + s

            # move the data into numpy arrays
            x = np.array([k for k in magnetisation.keys()])
            magnetisation = np.array([v for k, v in magnetisation.items()])

            result = lmfit.minimize(bloch_domain_wall_residual, fit_params, args=(x,),
                kws={'mx': magnetisation[:,0], 'my': magnetisation[:,1], 'mz': magnetisation[:,2]})

            print(f"{n:07} "
                  f"{result.params['center'].value:8.6f} {result.params['center'].stderr:8.6f} "
                  f"{result.params['height'].value:8.6f} {result.params['height'].stderr:8.6f} "
                  f"{result.params['width'].value:8.6f} {result.params['width'].stderr:8.6f}", file=outfile)


            if args.print_fits:
                print(lmfit.fit_report(result))

                plt.plot(x, magnetisation[:,0], label='$M_x$')
                plt.plot(x, magnetisation[:,1], label='$M_y$')
                plt.plot(x, magnetisation[:,2], label='$M_z$')

                plt.plot(x, bloch_domain_wall_mx(x, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_x$ fit')
                plt.plot(x, bloch_domain_wall_my(x, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_y$ fit')
                plt.plot(x, bloch_domain_wall_mz(x, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_z$ fit')


                plt.xlabel(r'$x$ (nm)')
                plt.ylabel(r'$M_{x,y,z}$ ($\mu_{B}$)')

                plt.title(filename)

                plt.legend()
                plt.savefig(f'{os.path.splitext(filename)[0]}.pdf')
                plt.close()
        except OSError:
            break

