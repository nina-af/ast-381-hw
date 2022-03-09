import matplotlib.pyplot as plt
import numpy as np
import os

from hyperion.model import ModelOutput
from hyperion.util.constants import pc

# Parse number of photons, grid resolution from command line.
n_photons = float(sys.argv[1])
n_r       = int(sys.argv[2])
n_theta   = int(sys.argv[3])
n_phi     = int(sys.argv[4])

exp_p = int(np.log10(n_photons))
pre_p = int(n_photons / 10**exp_p)

# Directory in which to store images.
im_dir = sys.argv[5]

# Model filename.
fname_in  = 'yso_{0:d}e{1:d}_{2:d}_{3:d}_{4:d}.rtout'.format(pre_p, exp_p, 
                                                             n_r, n_theta, n_phi)

mo = ModelOutput(fname_in)

# Viewing angles.
inclinations = np.linspace(0.0, 90.0, 10)

for i, inc in enumerate(inclinations):
    
    # Set up figure.
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(1, 1, 1)
    
    # Total SED.
    sed = mo.get_sed(aperture=-1, distance=300.0*pc, inclination=i)
    ax.loglog(sed.wav, sed.val, color='black', lw=3, alpha=0.5, label='all components')
    
    # Direct stellar photons.
    sed = mo.get_sed(aperture=-1, distance=300.0*pc, inclination=i,
                     component='source_emit')
    ax.loglog(sed.wav, sed.val, color='blue', label='direct stellar photons')
    
    # Scattered stellar photons.
    sed = mo.get_sed(aperture=-1, distance=300.0*pc, inclination=i,
                     component='source_scat')
    ax.loglog(sed.wav, sed.val, color='teal', label='scattered stellar photons')
    
    # Direct dust photons.
    sed = mo.get_sed(aperture=-1, distance=300.0*pc, inclination=i,
                     component='dust_emit')
    ax.loglog(sed.wav, sed.val, color='red', label='direct dust photons')
    
    # Scattered dust photons.
    sed = mo.get_sed(aperture=-1, distance=300.0*pc, inclination=i,
                     component='dust_scat')
    ax.loglog(sed.wav, sed.val, color='orange', label='scattered dust photons')
    
    ax.set_xlim(0.03, 2000.0)
    ax.set_ylim(2.0e-15, 1.0e-8)
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$\lambda F_\lambda}$ [erg/cm$^2/s$]')
    ax.set_title('SED components (inclination = {0:d})'.format(int(inc)))
    ax.legend()
    
    fstr = 'yso_{0:d}e{1:d}_{2:d}_{3:d}_{4:d}_inc_{5:d}_sed.png'.format(pre_p, exp_p, 
                                                                        n_r, n_theta, n_phi,
                                                                        int(inc))
    
    fname_out = os.path.join(im_dir, fstr)
    fig.savefig(fname_out, bbox_inches='tight')
