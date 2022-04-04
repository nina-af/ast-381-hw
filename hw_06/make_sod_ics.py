#!/usr/bin/env python

'''
Command-line input: 
- number of grid cells in 1D.
- method: 0 = MFM (equal particle masses), 1 = MFV (equal particle spacing).
- following make_ics.py in Gizmo /scripts.
'''

import numpy as np
import h5py as h5py
import sys

# Parse number of photons, grid resolution, box size from command line.
N = int(sys.argv[1])       # HW 2: N = 200, cfl = 0.5.
method = int(sys.argv[2])  # 0 = MFM, 1 = MFV.

L = 2.0  # cm

# Output filename.
if method == 0:
    fname = 'sod_shock_N{0:d}_mfm_ics.hdf5'.format(N)
else:
    fname = 'sod_shock_N{0:d}_mfv_ics.hdf5'.format(N)

rho_1 = 1.0         # Density for x < 0.75.
rho_2 = 0.125       # Density for x > 0.75.
P_1   = 1.0         # Pressure for x < 0.75.
P_2   = 0.1         # Pressure for x > 0.75.
x_0   = 0.75        # Initial contact discontinuity.

gamma_eos = 1.4
    
# MFM: equal particle masses.
if method == 0:
    m_1   = rho_1 * x_0
    m_2   = rho_2 * (L - x_0)
    m_tot = m_1 + m_2
    dm    = m_tot / N

    # Set all particles to have equal mass.
    m_g = dm * np.ones(N)
    
    N_1 = int(round(N * m_1 / (m_1 + m_2)))  # Number of particles with rho_1.
    N_2 = int(round(N * m_2 / (m_1 + m_2)))  # Number of particles with rho_2.
    
    # Initial particle spacing.
    dx_1 = x_0 / N_1; dx_2 = (L - x_0) / N_2
    
    x_1 = np.linspace(0.0, x_0, num=N_1, endpoint=False); x_1 = 0.5 * dx_1 + x_1
    x_2 = np.linspace(x_0, L, num=N_2, endpoint=False);   x_2 = 0.5 * dx_2 + x_2
    x_g = np.append(x_1, x_2); y_g = 0.0 * x_g; z_g = 0.0 * x_g
    
    uv_g = np.ones(N)
    uv_g[0:N_1] = P_1 / ((gamma_eos - 1.0) * rho_1)
    uv_g[N_1:]  = P_2 / ((gamma_eos - 1.0) * rho_2)
    
# MFV: equal particle spacing.
else:
    
    # Equal spacing between particles.
    x_g = np.linspace(0, L, num=N, endpoint=False)
    x_g = 0.5 * (x_g[1] - x_g[0]) + x_g; y_g = 0.0 * x_g; z_g = 0.0 * x_g
    x_i = int((x_0 / L) * N)
    
    dm_1 = (rho_1 * x_0) / x_i; dm_2 = (rho_2 * (L - x_0)) / (N - x_i)
    m_g = np.ones(N); m_g[0:x_i] = dm_1; m_g[x_i:] = dm_2
    
    uv_g = np.ones(N)
    uv_g[0:x_i] = P_1 / ((gamma_eos - 1.0) * rho_1)
    uv_g[x_i:]  = P_2 / ((gamma_eos - 1.0) * rho_2)
   
# Set initial velocities equal to zero.
vx_g = 0.0 * x_g; vy_g = 0.0 * x_g; vz_g = 0.0 * x_g

# Set the gas IDs: here a simple integer list.
id_g = np.arange(1, N + 1)

# Write all IC data to hdf5 file.
file = h5py.File(fname, 'w') 
    
npart = np.array([N, 0, 0, 0, 0, 0])
    
h = file.create_group("Header");

h.attrs['BoxSize'] = L
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
q = np.zeros((N, 3)); q[:, 0] = x_g; q[:, 1] = y_g; q[:, 2] = z_g
p.create_dataset("Coordinates", data=q)

q = np.zeros((N, 3)); q[:, 0] = vx_g; q[:, 1] = vy_g; q[:, 2] = vz_g
p.create_dataset("Velocities", data=q)
p.create_dataset("ParticleIDs", data=id_g)
p.create_dataset("Masses", data=m_g)

p.create_dataset("InternalEnergy", data=uv_g)
    
# Close the HDF5 file, which saves these outputs
file.close()

