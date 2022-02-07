#!/usr/bin/env python3

# 2022-01-27 Joseph Barker (j.barker@leeds.ac.uk)
# This python script is for fitting Bloch domain walls in JAMS h5 output files.
# For a given simulation name it read each spin output h5 file and fit the mx,my,mz
# data with a Bloch wall profile, where it is assumed the Bloch wall is unconstrained,
# and the magnetisation at the left/right sides are in the -/+z directions.
#
# Requirements
# ------------
# - numpy
# - matplotlib
# - h5py
# - lmfit
#
# Usage
# -----
# usage: bloch_domain_wall_fitting.py [-h] -i INFILE [-o [OUTFILE]] [-p] [-f {mx,my,mz} [{mx,my,mz} ...]]
#
# optional arguments:
#   -h, --help                         show this help message and exit
#   -i INFILE, --input INFILE          input h5 file
# -o [OUTFILE], --output [OUTFILE]
# -p, --print-pdf                      output pdf of the fitting
# -f {mx,my,mz} [{mx,my,mz} ...], --fit-profiles {mx,my,mz} [{mx,my,mz} ...]
#                          select magnetisation profiles to include in the fit

import os
import re
import sys
import h5py
import lmfit
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    return height/np.cosh(np.pi*(x-center)/width)


def bloch_domain_wall_mz(x, center, height, width):
    """
    Bloch domain wall z-magnetisation profile

    x:       x coordinate
    center: domain wall center
    height: domain wall height (magnetization along z)
    width:  domain wall width
    """
    return height*np.tanh(np.pi*(x-center)/width)


def bloch_domain_wall_residual(params, x, mx=None, my=None, mz=None):
    """
    Residual function for use with lmfit.minimize to fit a Bloch domain wall
    using mx, my and mz data.

    Although the bloch_domain_wall_mx is zero, but including it in the residual we get a better
    estimate of the errors because they include information about how far the
    x profile has deviated from a flat line.
    """

    resid_x = np.empty((0))
    resid_y = np.empty((0))
    resid_z = np.empty((0))

    if mx is not None:
        model_x = bloch_domain_wall_mx(x, params['center'], params['height'], params['width'])
        resid_x = mx - model_x

    if my is not None:
        model_y = bloch_domain_wall_my(x, params['center'], params['height'], params['width'])
        resid_y = my - model_y

    if mz is not None:
        model_z = bloch_domain_wall_mz(x, params['center'], params['height'], params['width'])
        resid_z = mz - model_z

    return np.concatenate((resid_x, resid_y, resid_z))


def get_simulation_name(filename):
    """
    Determines the simulation name based on an input h5 file
    """

    match = re.match(SPINFILE_REGEX, filename)
    return match.group(1)

def get_output_num(filename):
    """
    Determines h5 output sequence number
    """

    match = re.match(SPINFILE_REGEX, filename)
    return match.group(2)

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", dest="infile", required=True,
                    help="input h5 file")

parser.add_argument("-o", "--output", nargs='?', dest="outfile",
                    type=argparse.FileType('w'), default=sys.stdout)

parser.add_argument("-p", "--print-pdf", dest="print_pdf", action='store_true',
                    help="output pdf of the fitting")

parser.add_argument("-f", "--fit-profiles", nargs='+', dest="fit_profiles",
                    choices=['mx', 'my', 'mz'], default=['mz'],
                    help="select magnetisation profiles to include in the fit")

args = parser.parse_args()



with h5py.File(args.infile, "r") as f:
    print('time_ps dw_center_nm dw_center_nm_stderr dw_mz_muB dw_mz_muB_stderr dw_width_nm dw_width_nm_stderr', file=args.outfile)

    positions = np.array(f['/jams/monitors/magnetisation_layers/layer_positions'])

    for name, obj in f['/jams/monitors/magnetisation_layers'].items():
        # We only want the groups with names like 000000000 for the time series data
        if not isinstance(obj, h5py.Group) or 'time' not in obj.attrs:
            continue


        fit_params = lmfit.Parameters()
        fit_params.add('center', value = np.mean(positions))
        fit_params.add('height', value = 1000.0)
        fit_params.add('width',  value = 10.0)

        time = float(obj.attrs['time'])

        magnetisation = np.array(obj['layer_magnetisation'])

        fit_data = {}

        if 'mx' in args.fit_profiles:
            fit_data['mx'] = magnetisation[:,0]

        if 'my' in args.fit_profiles:
            fit_data['my'] = magnetisation[:,1]

        if 'mz' in args.fit_profiles:
            fit_data['mz'] = magnetisation[:,2]

        result = lmfit.minimize(bloch_domain_wall_residual, fit_params, args=(positions,),
                                kws=fit_data)

        print(f"{time:8.6f} "
              f"{result.params['center'].value:8.6f} {result.params['center'].stderr:8.6f} "
              f"{result.params['height'].value:8.6f} {result.params['height'].stderr:8.6f} "
              f"{result.params['width'].value:8.6f} {result.params['width'].stderr:8.6f}", file=args.outfile)
        

        if args.print_pdf:
            plt.plot(positions, magnetisation[:,0], label='$M_x$')
            plt.plot(positions, magnetisation[:,1], label='$M_y$')
            plt.plot(positions, magnetisation[:,2], label='$M_z$')
        
            if 'mx' in args.fit_profiles:
                plt.plot(positions, bloch_domain_wall_mx(positions, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_x$ fit')
        
            if 'my' in args.fit_profiles:
                plt.plot(positions, bloch_domain_wall_my(positions, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_y$ fit')
        
            if 'mz' in args.fit_profiles:
                plt.plot(positions, bloch_domain_wall_mz(positions, result.params['center'], result.params['height'], result.params['width']), '--', label='$M_z$ fit')
        
        
            plt.xlabel(r'$x$ (nm)')
            plt.ylabel(r'$M_{x,y,z}$ ($\mu_{B}$)')
        
            plt.title(f'{args.infile}_{name}')
        
            plt.legend()
            plt.savefig(f'{args.infile}_{name}.pdf')
            plt.close()
            
