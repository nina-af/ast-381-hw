#!/usr/bin/env python

'''
Command-line input: 
- number of grid cells in 1D.
- box lenght [cm].
- following make_ics.py in Gizmo /scripts.
'''

import numpy as np
import h5py as h5py
import sys

# Parse number of photons, grid resolution, box size from command line.
N_1D = int(sys.argv[1])
Lbox = float(sys.argv[2])

# Output filename.
fname = 'mhd_blast_n{0:d}_ics.hdf5'.format(N_1D)

DIMS        = 2            # 2D grid.
R_0         = 0.1 * Lbox   # Circle of radius 0.1 * Lbox cm.
rho_desired = 1.0          # Box average initial gas density [g / cm^2].
Pi_desired  = 10.0         # Pressure inside circle of radius R_0.
Po_desired  = 0.1          # Pressure outside circle of radius R_0.
    
gamma_eos = 5.0 / 3.0  # Polytropic index of ideal equation of state the run will assume.
    
B_desired = 1.0 / np.sqrt(2.0)  # (Bx, By, Bz) = (1/sqrt(2), 1/sqrt(2), 0).

# Make a regular 1D grid for particle locations (with N_1D elements and unit length).
x0 = np.arange(-0.5, 0.5, 1./N_1D); x0 += 0.5 * (0.5 - x0[-1])
    
# Now extend that to a full lattice in DIMS dimensions
xv_g, yv_g = np.meshgrid(x0, x0, sparse=False, indexing='xy'); zv_g = 0.0 * xv_g

# Mask of circle of radius R_0 centered at (0.0, 0.0).
r_vals = np.sqrt(xv_g**2 + yv_g**2)
r0_vals = np.full((N_1D, N_1D), R_0, dtype=float)
r_mask = np.less(r_vals, r0_vals)

# Gas particle number is the lattice size: this should be the gas particle number
Ngas = xv_g.size
    
# Flatten the vectors (since our ICs should be in vector, not matrix format).
xv_g = xv_g.flatten() * Lbox
yv_g = yv_g.flatten() * Lbox
zv_g = zv_g.flatten() * Lbox

# Flatten r mask as well.
r_mask = r_mask.flatten()

# Set the initial velocity in x/y/z directions (here zero).
vx_g = 0.0 * xv_g; vy_g = 0.0 * xv_g; vz_g = 0.0 * xv_g
    
# Set the initial magnetic field in x/y/z directions.
bx_g = B_desired * np.ones(len(xv_g))
by_g = B_desired * np.ones(len(xv_g))
bz_g = np.zeros(len(xv_g))
    
# Set the particle masses. Here we set it to be a list the same length, with all the same 
# mass - since their space-density is uniform this gives a uniform density, at the desired 
# value.
mv_g  = rho_desired / ((1.0 * Ngas) / (Lbox * Lbox)) + 0.0 * xv_g

# Set the initial internal energy per unit mass. Recall GIZMO uses this as the initial 
# 'temperature' variable.
uv_g = np.ones(len(xv_g))

uv_g[r_mask]            = Pi_desired / ((gamma_eos - 1.0) * rho_desired)
uv_g[np.invert(r_mask)] = Po_desired / ((gamma_eos - 1.0) * rho_desired)

# Set the gas IDs: here a simple integer list.
id_g = np.arange(1, Ngas + 1)

# Write all IC data to hdf5 file.

file = h5py.File(fname, 'w') 
    
npart = np.array([Ngas, 0, 0, 0, 0, 0])
    
h = file.create_group("Header");

h.attrs['BoxSize'] = Lbox
h.attrs['Flag_Cooling'] = 0
h.attrs['Flag_Feedback'] = 0
h.attrs['Flag_IC_Info'] = 0
h.attrs['Flag_Metals'] = 0
h.attrs['Flag_Sfr'] = 0 
h.attrs['Flag_StellarAge'] = 0 

h.attrs['NumPart_ThisFile'] = npart;
h.attrs['NumPart_Total'] = npart; 
h.attrs['NumPart_Total_HighWord'] = 0 * npart;

h.attrs['HubbleParam'] = 1.0
h.attrs['Omega0'] = 0.0
h.attrs['OmegaLambda'] = 0.0
h.attrs['Redshift'] = 0.0

h.attrs['MassTable'] = np.zeros(6); 
h.attrs['Time'] = 0.0; 
h.attrs['NumFilesPerSnapshot'] = 1; 
h.attrs['Flag_DoublePrecision'] = 1;
    
p = file.create_group("PartType0")
q = np.zeros((Ngas, 3)); q[:, 0] = xv_g; q[:, 1] = yv_g; q[:, 2] = zv_g
p.create_dataset("Coordinates", data=q)

q = np.zeros((Ngas, 3)); q[:, 0] = vx_g; q[:, 1] = vy_g; q[:, 2] = vz_g
p.create_dataset("Velocities", data=q)
p.create_dataset("ParticleIDs", data=id_g)
p.create_dataset("Masses", data=mv_g)

p.create_dataset("InternalEnergy", data=uv_g)

q = np.zeros((Ngas, 3)); q[:, 0] = bx_g; q[:, 1] = by_g; q[:, 2] = bz_g
p.create_dataset("MagneticField", data=q)
    
# Close the HDF5 file, which saves these outputs
file.close()

