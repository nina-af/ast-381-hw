import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to compute f(z_i) for each z_i.
def function_values_comoving(start, stop, n, omega_m, omega_l):

    z = np.linspace(start, stop, num=n)
    f1 = omega_m * (1.0 + z)**3
    f2 = (1.0 - omega_m - omega_l) * (1.0 + z)**2
    f = (f1 + f2 + omega_l) ** (-0.5)

    return f

# Function to numerically integrate using the trapezoid rule.
def trapezoid_rule_comoving(start, stop, n, omega_m, omega_l):

    dz = (stop - start) / (n - 1)
    f_vals = function_values_comoving(start, stop, n, omega_m, omega_l)

    integral = (dz / 2.0) * (f_vals[0] + f_vals[n - 1] + 2.0 * np.sum(f_vals[1:(n - 1)]))

    return integral

# Comoving distance to redshift z = 2.
om = 0.3
ol = 0.7
n = 100000

d2 = 3000.0 * trapezoid_rule_comoving(0.0, 2.0, n, om, ol)
print('Comoving distance to z = 2: {0:.6g} Mpc h^-1'.format(d2))
print('Comoving distance to z = 2: {0:.6g} Mpc'.format(d2 * 0.70**(-1)))

# Comoving distance from z = 0 to z = 10.
z_vals = np.linspace(0.0, 10.0, num=100)
d_vals = np.zeros(100)

for i in range(len(z_vals)):
    d_vals[i] = 3000.0 * trapezoid_rule_comoving(0.0, z_vals[i], n, om, ol)

# Plot comoving distance as a function of redshift.
plt.rcParams["figure.figsize"] = (12,6)

plt.plot(z_vals, d_vals, '-o', linewidth=3)

plt.grid(True)
plt.xlabel('$z$')
plt.ylabel('Comoving distance (Mpc $h^{-1}$)')
plt.title('Comoving distance vs. redshift ($\Omega_m = 0.3$, $\Omega_{\Lambda} = 0.7$)')

plt.show()

# Plot comoving distance as a function of redshift.
plt.rcParams["figure.figsize"] = (12 ,6)

plt.plot(z_vals, (d_vals * 0.70**(-1)), '-o', linewidth=3)

plt.grid(True)
plt.xlabel('$z$')
plt.ylabel('Comoving distance (Mpc)')
plt.title('Comoving distance vs. redshift ($\Omega_m = 0.3$, $\Omega_{\Lambda} = 0.7$, $h = 0.70$)')

plt.show()
