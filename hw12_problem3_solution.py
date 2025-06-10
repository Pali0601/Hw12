import numpy as np
import matplotlib.pyplot as plt

r_min, r_max = 0.5, 1.0
theta_min, theta_max = 0, np.pi / 3
dr = 0.1
dtheta = np.pi / 30

nr = int((r_max - r_min) / dr)
ntheta = int((theta_max - theta_min) / dtheta)

r = np.linspace(r_min, r_max, nr + 1)
theta = np.linspace(theta_min, theta_max, ntheta + 1)

T = np.zeros((nr + 1, ntheta + 1))
T[:, 0] = 0
T[:, -1] = 0
T[0, :] = 50
T[-1, :] = 100

def solve_polar_laplace_fixed(T, max_iter=10000, tol=1e-5):
    for iteration in range(max_iter):
        max_diff = 0
        for i in range(1, nr):
            ri = r[i]
            for j in range(1, ntheta):
                T_new = (
                    (1 / dr**2 - 1 / (2 * ri * dr)) * T[i - 1, j] +
                    (1 / dr**2 + 1 / (2 * ri * dr)) * T[i + 1, j] +
                    (1 / (ri**2 * dtheta**2)) * (T[i, j - 1] + T[i, j + 1])
                ) / (2 / dr**2 + 2 / (ri**2 * dtheta**2))
                diff = abs(T_new - T[i, j])
                max_diff = max(max_diff, diff)
                T[i, j] = T_new
        if max_diff < tol:
            break
    return T, iteration

T_sol3, iterations = solve_polar_laplace_fixed(T.copy())

R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, T_sol3.T, 50, cmap='inferno')
plt.colorbar(label='T(r, θ)')
plt.title(f'Problem 3: Corrected Polar Laplace (converged in {iterations} iterations)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()
