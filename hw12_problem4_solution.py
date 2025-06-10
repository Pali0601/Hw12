import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_min, x_max = 0, 1
t_min, t_max = 0, 1
dx = dt = 0.1
nx = int((x_max - x_min) / dx)
nt = int((t_max - t_min) / dt)

x = np.linspace(x_min, x_max, nx + 1)
t = np.linspace(t_min, t_max, nt + 1)

alpha = (dt / dx)**2
p = np.zeros((nx + 1, nt + 1))

p[:, 0] = np.cos(2 * np.pi * x)
p[1:-1, 1] = (
    p[1:-1, 0] + dt * 2 * np.pi * np.sin(2 * np.pi * x[1:-1]) +
    0.5 * alpha * (p[2:, 0] - 2 * p[1:-1, 0] + p[0:-2, 0])
)

p[0, :] = 1
p[-1, :] = 2

for j in range(1, nt):
    for i in range(1, nx):
        p[i, j+1] = 2 * p[i, j] - p[i, j-1] + alpha * (p[i+1, j] - 2 * p[i, j] + p[i-1, j])

X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, p.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('p(x, t)')
ax.set_title('Problem 4: Wave Equation Solution')
plt.tight_layout()
plt.show()
