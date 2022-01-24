import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate

# Function to compute f(x_i) = x_i^(-1.5) for each x_i.
def function_values(start, stop, num_points):

    x_vals = np.linspace(start, stop, num=num_points)
    y_vals = np.ones(num_points) * np.power(x_vals, -1.5)

    return y_vals

# Function to compute the relative error.
def frac_err(int_exact, int_num):

    return abs(int_exact - int_num) / int_exact

# SciPy trapezoid rule.
N_vals_trap = np.asarray([5, 10, 20, 30, 43, 44, 50, 100, 200, 422, 423, 500, 
                          800, 1000, 1200, 1500, 2000, 5000, 8000, 10000])

# Simpson's rule (number of samples must be odd).
N_vals_simp = np.asarray([5, 11, 13, 21, 37, 51, 101, 201, 501, 801, 1001, 1201, 1501, 2001])

# Romberg integration (number of samples must be 2^k + 1).
N_vals_romb = np.asarray([5, 9, 17, 33, 65, 129, 257])

frac_err_vals_trap = np.zeros(len(N_vals_trap))
frac_err_vals_simp = np.zeros(len(N_vals_simp))
frac_err_vals_romb = np.zeros(len(N_vals_romb))

# Integration limits.
a = 1.0
b = 5.0

# Exact integral.
I_exact = 0.4 * (5.0 - np.sqrt(5.0))

print('I_exact = {0:.10f}'.format(I_exact))

print('\nscipy.integrate.trapz:')
for i in range(len(N_vals_trap)):

    dx = (b - a) / (N_vals_trap[i] - 1)
    f_vals = function_values(a, b, N_vals_trap[i])

    I_num = scipy.integrate.trapz(f_vals, dx=dx)
    frac_err_vals_trap[i] = frac_err(I_exact, I_num)

    print('N = {0:<6d}\tI_num = {1:.15f}\tfrac_err = {2:.15f}'.format(N_vals_trap[i],
                                                                      I_num, frac_err_vals_trap[i]))

print('\nscipy.integrate.simps:')
for i in range(len(N_vals_simp)):

    dx = (b - a) / (N_vals_simp[i] - 1)
    f_vals = function_values(a, b, N_vals_simp[i])

    I_num = scipy.integrate.simps(f_vals, dx=dx)
    frac_err_vals_simp[i] = frac_err(I_exact, I_num)

    print('N = {0:<6d}\tI_num = {1:.15f}\tfrac_err = {2:.15f}'.format(N_vals_simp[i],
                                                                      I_num, frac_err_vals_simp[i]))

print('\nscipy.integrate.romb:')
for i in range(len(N_vals_romb)):

    dx = (b - a) / (N_vals_romb[i] - 1)
    f_vals = function_values(a, b, N_vals_romb[i])

    I_num = scipy.integrate.romb(f_vals, dx)
    frac_err_vals_romb[i] = frac_err(I_exact, I_num)

    print('N = {0:<6d}\tI_num = {1:.15f}\tfrac_err = {2:.15f}'.format(N_vals_romb[i],
                                                                      I_num, frac_err_vals_romb[i]))

# Convergence plots
fit_trap = np.polyfit(np.log10(N_vals_trap), np.log10(frac_err_vals_trap), 1)
fit_simp = np.polyfit(np.log10(N_vals_simp), np.log10(frac_err_vals_simp), 1)

label_trap = "Trapezoid rule (slope = {0:.3f})".format(fit_trap[0])
label_simp = "Simpson's rule (slope = {0:.3f})".format(fit_simp[0])
label_romb = "Romberg integration"

plt.rcParams["figure.figsize"] = (12,6)

plt.plot(N_vals_trap, frac_err_vals_trap, '-o', label=label_trap, linewidth=3)
plt.plot(N_vals_simp, frac_err_vals_simp, '-o', label=label_simp, linewidth=3)
plt.plot(N_vals_romb, frac_err_vals_romb, '-o', label=label_romb, linewidth=3)

# Plot horizontal lines where fractional error is 1.0e-3, 1.0e-5.
plt.axhline(y=1.0e-3, color='r', linestyle='-')
plt.axhline(y=1.0e-5, color='r', linestyle='-')

# Plot vertical lines where rectangle rule achieves sub-1.0e-3, 1.0e-5 accuracy.
plt.axvline(x=44, color='tab:blue', linestyle='--')
plt.axvline(x=423, color='tab:blue', linestyle='--')

# Plot vertical lines where trapezoid rule achieves sub-1.0e-3, 1.0e-5 accuracy.
plt.axvline(x=13, color='tab:orange', linestyle='--')
plt.axvline(x=51, color='tab:orange', linestyle='--')

# Plot vertical lines where trapezoid rule achieves sub-1.0e-3, 1.0e-5 accuracy.
plt.axvline(x=17, color='g', linestyle='--')
plt.axvline(x=33, color='g', linestyle='--')

plt.grid(True)
plt.xlabel('$N_{points}$')
plt.ylabel('Fractional error')
plt.title('Convergence plot (scipy.integrate routines)')
plt.xscale('log')
plt.yscale('log')
plt.legend(shadow=True)
plt.show()
