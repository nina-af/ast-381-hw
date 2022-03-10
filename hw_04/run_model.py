#!/usr/bin/env python

'''
Command-line arguments: 
- number of photons
- grid: n_r, n_theta, n_phi
'''

import numpy as np
import sys

from hyperion.model import AnalyticalYSOModel
from hyperion.util.constants import rsun, lsun, msun, au, yr, c

# Parse number of photons, grid resolution from command line.
n_photons = float(sys.argv[1])
n_r       = int(sys.argv[2])
n_theta   = int(sys.argv[3])
n_phi     = int(sys.argv[4])

# Dust model for all.
fname_dust = '/home/nina/hyperion-dust-0.1.0/dust_files/kmh94_3.1_full.hdf5'

# Initialize the model.
m = AnalyticalYSOModel()

# Stellar luminosity, radius, temperature.
m.star.luminosity  = 5.0 * lsun
m.star.radius      = 2.0 * rsun
m.star.temperature = 6200.0

# Flared disk.
disk = m.add_flared_disk()
disk.mass = 0.01 * msun
disk.rmin = 10 * rsun
disk.rmax = 200 * au
disk.p    = -1.0
disk.beta = 1.25
disk.r_0  = m.star.radius
disk.h_0  = 0.01 * disk.r_0
disk.dust = fname_dust

# Envelope.
envelope = m.add_power_law_envelope()
envelope.mass  = 0.4 * msun
envelope.rmin  = 200 * au
envelope.rmax  = 10000 * au
envelope.power = -2
envelope.r_0   = disk.rmax
envelope.dust  = fname_dust

# Enable raytracing.
m.set_raytracing(True)

# For high densities, enable modified random walk.
m.set_mrw(True, gamma=2.)

# Output specific energy and density.
m.conf.output.output_specific_energy = 'last'
m.conf.output.output_density = 'last'

# Grid.
m.set_spherical_polar_grid_auto(n_r, n_theta, n_phi)

# Set up SED.
sed = m.add_peeled_images(sed=True, image=False)
sed.set_viewing_angles(np.linspace(0.0, 90.0, 10), np.repeat(45.0, 10))
sed.set_wavelength_range(150, 0.02, 2000.0)
sed.set_track_origin('basic')

# Set number of photons.
m.set_n_photons(initial=n_photons, imaging=n_photons, 
                raytracing_sources=n_photons, raytracing_dust=n_photons)

# Set number of iterations and convergence criterion.
m.set_n_initial_iterations(10)
m.set_convergence(True, percentile=99.0, absolute=2.0, relative=1.1)

# Write out file.
exp_p = int(np.log10(n_photons))
pre_p = int(n_photons / 10**exp_p)
fname_write = 'yso_{0:d}e{1:d}_{2:d}_{3:d}_{4:d}.rtin'.format(pre_p, exp_p, n_r, n_theta, n_phi)
fname_run   = 'yso_{0:d}e{1:d}_{2:d}_{3:d}_{4:d}.rtout'.format(pre_p, exp_p, n_r, n_theta, n_phi)

# Write and run model.
m.write(fname_write)
m.run(fname_run, mpi=False)
