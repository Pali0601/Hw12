import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
h = k = 0.1 * pi
nx = int(pi / h)
ny = int((pi / 2) / k)
alpha = (h / k)**2

x = np.linspace(0, pi, nx + 1)
y = np.linspace(0, pi / 2, ny + 1)
u = np.zeros((nx + 1, ny + 1))

u[0, :] = np.cos(y)
u[-1, :] = -np.cos(y)
u[:, 0] = np.cos(x)
u[:, -1] = 0

def solve_laplace(u, alpha, tol=1e-5, max_iter=10000):
    for iteration in range(max_iter):
        max_diff = 0
        for i in range(1, nx):
            for j in range(1, ny):
                u_new = 1 / (2 * (1 + alpha)) * (
                    u[i+1, j] + u[i-1, j] + alpha * (u[i, j+1] + u[i, j-1])
                )
                diff = abs(u_new - u[i, j])
                max_diff = max(max_diff, diff)
                u[i, j] = u_new
        if max_diff < tol:
            break
    return u, iteration

u_result, iterations = solve_laplace(u.copy(), alpha)

X, Y = np.meshgrid(x, y)
plt.figure(figsize=(8, 5))
plt.contourf(X, Y, u_result.T, 50, cmap='coolwarm')
plt.colorbar(label='u(x, y)')
plt.title(f'Laplace Equation Solution (converged in {iterations} iterations)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
