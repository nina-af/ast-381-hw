import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to compute f(x_i) = x_i^(-1.5) for each x_i.
def function_values(start, stop, num_points):

    x_vals = np.linspace(start, stop, num=num_points)
    y_vals = np.ones(num_points) * np.power(x_vals, -1.5)

    return y_vals

# Function to numerically integrate using the rectangle rule.
def rectangle_rule(start, stop, num_points):

    dx = (stop - start) / (num_points - 1)              # Spacing between points.
    y_vals = function_values(start, stop, num_points)

    integral = dx * np.sum(y_vals[0:(num_points - 2)])  # Exclude last point (x_N = b).

    return integral

# Function to numerically integrate using the trapezoid rule.
def trapezoid_rule(start, stop, num_points):

    dx = (stop - start) / (num_points - 1)
    y_vals = function_values(start, stop, num_points)

    integral = (dx / 2.0) * (y_vals[0] + y_vals[num_points - 1] + 2.0 * np.sum(y_vals[1:(num_points - 1)]))

    return integral

# Function to compute the relative error.
def frac_err(int_exact, int_num):

    return abs(int_exact - int_num) / int_exact

# Integration limits.
a = 1.0
b = 5.0

# Exact integral.
I_exact = 0.4 * (5.0 - np.sqrt(5.0))

print('I_exact = {0:.10f}'.format(I_exact))

N_vals_r        = np.asarray([5, 10, 20, 50, 100, 200, 500, 800, 1000, 1200, 
                              1325, 1326, 
                              1500, 1800, 2000, 5000,
                              8000, 10000, 20000, 50000, 80000, 
                              100000, 120000, 
                              132363, 132364,
                              150000, 180000, 200000])
N_vals_t        = np.asarray([5, 10, 20, 
                              30, 43, 44,
                              50, 100, 200, 
                              422, 423,
                              500, 800, 1000, 1200, 
                              1500, 2000, 5000, 8000, 10000])

frac_err_vals_r = np.zeros(len(N_vals_r))
frac_err_vals_t = np.zeros(len(N_vals_t))

print('\nRectangle rule:')
for i in range(len(N_vals_r)):
    
    I_num = rectangle_rule(a, b, N_vals_r[i])    
    frac_err_vals_r[i] = frac_err(I_exact, I_num)
    
    print('N = {0:<6d}\tI_num = {1:.10f}\tfrac_err = {2:.10f}'.format(N_vals_r[i], I_num, frac_err_vals_r[i]))
    
print('\nTrapezoid rule:')
for i in range(len(N_vals_t)):
    
    I_num = trapezoid_rule(a, b, N_vals_t[i])    
    frac_err_vals_t[i] = frac_err(I_exact, I_num)
    
    print('N = {0:<6d}\tI_num = {1:.10f}\tfrac_err = {2:.10f}'.format(N_vals_t[i], I_num, frac_err_vals_t[i]))

# Convergence plots
fit_r = np.polyfit(np.log10(N_vals_r), np.log10(frac_err_vals_r), 1)
fit_t = np.polyfit(np.log10(N_vals_t), np.log10(frac_err_vals_t), 1)

label_r = "Rectangle rule (slope = {0:.3f})".format(fit_r[0])
label_t = "Trapezoid rule (slope = {0:.3f})".format(fit_t[0])

plt.rcParams["figure.figsize"] = (12,6)

plt.plot(N_vals_r, frac_err_vals_r, '-o', label=label_r, linewidth=3)
plt.plot(N_vals_t, frac_err_vals_t, '-o', label=label_t, linewidth=3)

# Plot horizontal lines where fractional error is 1.0e-3, 1.0e-5.
plt.axhline(y=1.0e-3, color='r', linestyle='-')
plt.axhline(y=1.0e-5, color='r', linestyle='-')

# Plot vertical lines where rectangle rule achieves sub-1.0e-3, 1.0e-5 accuracy.
plt.axvline(x=1326, color='tab:blue', linestyle='--')
plt.axvline(x=132364, color='tab:blue', linestyle='--')

# Plot vertical lines where trapezoid rule achieves sub-1.0e-3, 1.0e-5 accuracy.
plt.axvline(x=44, color='tab:orange', linestyle='--')
plt.axvline(x=423, color='tab:orange', linestyle='--')

plt.grid(True)
plt.xlabel('$N_{points}$')
plt.ylabel('Fractional error')
plt.title('Convergence plot (Rectangle vs. Trapezoid Rule)')
plt.xscale('log')
plt.yscale('log')
plt.legend(shadow=True)
plt.show()
