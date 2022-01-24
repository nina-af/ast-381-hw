import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to return RHS of equations of motion for orbit. X, Y = position, velocity vectors.
def rhs(X, V):

    GM = 4.0*np.pi**2               # In units of AU, year, Msun
    r = np.sqrt(X[0]**2 + X[1]**2)  # Radius

    x_dot = V[0]
    y_dot = V[1]

    u_dot = -GM * X[0] / r**3
    v_dot = -GM * X[1] / r**3

    return x_dot, y_dot, u_dot, v_dot

# Function for first-order Euler integration.
def Euler_integration(dt):

    GM = 4.0*math.pi**2  # In units of AU, year, Msun

    # Semi-major axis (in AU) and eccentricity.
    a = 1.000001018
    e = 0.0167086

    # Initial conditions
    t = 0.0
    x = 0.0
    y = a * (1.0 - e)
    u = -np.sqrt((GM / a) * ((1.0 + e) / (1.0 - e)))
    v = 0.0

    # Store coordinates for plotting orbits.
    x_coords = [x]
    y_coords = [y]

    # Integrate for one orbit (t = 0 to t = 1).
    t_max = 1.0
    while t < t_max:

        if t + dt > t_max:
            dt = t_max - t

        x_dot, y_dot, u_dot, v_dot = rhs([x, y], [u, v])

        # Advance by one step (first-order Euler).
        x2 = x + dt * x_dot
        y2 = y + dt * y_dot
        u2 = u + dt * u_dot
        v2 = v + dt * v_dot

        t += dt

        # Store values.
        x_coords.append(x2)
        y_coords.append(y2)

        x = x2
        y = y2
        u = u2
        v = v2

    return x_coords, y_coords

# Function for fourth-order Runge-Kutta integration.
def RK4_integration(dt):

    GM = 4.0*math.pi**2  # In units of AU, year, Msun

    # Semi-major axis (in AU) and eccentricity.
    a = 1.000001018
    e = 0.0167086

    # Initial conditions
    t = 0.0
    x = 0.0
    y = a * (1.0 - e)
    u = -np.sqrt((GM / a) * ((1.0 + e) / (1.0 - e)))
    v = 0.0

    # Store coordinates
    x_coords = [x]
    y_coords = [y]

    # Integrate for one orbit (t = 0 to t = 1).
    t_max = 1.0
    while t < t_max:

        if t + dt > t_max:
            dt = t_max - t

        k1_x, k1_y, k1_u, k1_v = rhs([x, y], [u, v])
        k2_x, k2_y, k2_u, k2_v = rhs([x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y],
                                     [u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v])
        k3_x, k3_y, k3_u, k3_v = rhs([x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y],
                                     [u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v])
        k4_x, k4_y, k4_u, k4_v = rhs([x + dt * k3_x, y + dt * k3_y],
                                     [u + dt * k3_u, v + dt * k3_v])

        # Advance by one step (fourth-order Runge-Kutta).
        x2 = x + (dt / 6.0) * (k1_x + 2.0 * (k2_x + k3_x) + k4_x)
        y2 = y + (dt / 6.0) * (k1_y + 2.0 * (k2_y + k3_y) + k4_y)
        u2 = u + (dt / 6.0) * (k1_u + 2.0 * (k2_u + k3_u) + k4_u)
        v2 = v + (dt / 6.0) * (k1_v + 2.0 * (k2_v + k3_v) + k4_v)

        t += dt

        # Store values.
        x_coords.append(x2)
        y_coords.append(y2)

        x = x2
        y = y2
        u = u2
        v = v2

    return x_coords, y_coords

# Function to calculate fractional error in radius after one orbit.
def frac_error_radius(x_coords, y_coords):

    n = len(x_coords)

    x0 = x_coords[0]
    y0 = y_coords[0]
    xn = x_coords[n - 1]
    yn = y_coords[n - 1]

    r0 = np.sqrt(x0**2 + y0**2)
    rn = np.sqrt(xn**2 + yn**2)

    error = abs(r0 - rn) / r0

    return error

# Plot sample orbit.
N = 500
dt = 1.0 / (N - 1)

#x_coords, y_coords = Euler_integration(dt)
x_coords, y_coords = RK4_integration(dt)

plt.scatter(x_coords, y_coords)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect(1)
plt.title('Orbit')
plt.show()

# Compare first-order Euler and fourth-order Runge-Kutta for different step sizes.
n_vals = np.asarray([10, 20, 50, 100, 120, 150, 200, 500, 1000, 2000, 5000, 8000,
                     10000, 20000, 50000, 100000])

frac_err_eul = np.zeros(len(n_vals))
frac_err_rk4   = np.zeros(len(n_vals))

for i in range(len(n_vals)):

    dt = 1.0 / (n_vals[i] - 1)

    x_eul, y_eul = Euler_integration(dt)
    x_rk4, y_rk4 = RK4_integration(dt)

    frac_err_eul[i] = frac_error_radius(x_eul, y_eul)
    frac_err_rk4[i] = frac_error_radius(x_rk4, y_rk4)

# Convergence plots
fit_eul = np.polyfit(np.log10(n_vals), np.log10(frac_err_eul), 1)
fit_rk4 = np.polyfit(np.log10(n_vals)[0:8], np.log10(frac_err_rk4)[0:8], 1)

label_eul = "1st-order Euler (slope = {0:.3f})".format(fit_eul[0])
label_rk4 = "4th-order Runge-Kutta (slope = {0:.3f})".format(fit_rk4[0])

plt.rcParams["figure.figsize"] = (12,6)

plt.plot(n_vals, frac_err_eul, '-o', label=label_eul, linewidth=3)
plt.plot(n_vals[0:8], frac_err_rk4[0:8], '-o', label=label_rk4, linewidth=3)

plt.grid(True)
plt.xlabel('$N_{steps}$')
plt.ylabel('Fractional change in radius after one orbit')
plt.title('Convergence plot (Euler vs. Runge-Kutta)')
plt.xscale('log')
plt.yscale('log')
plt.legend(shadow=True)
plt.show()
