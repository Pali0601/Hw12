import numpy as np
import matplotlib.pyplot as plt

K = 0.1
r_min, r_max = 0.5, 1.0
t_max = 10
dr = 0.1
dt = 0.5
alpha = dt / (K * dr**2)

nr = int((r_max - r_min) / dr)
nt = int(t_max / dt)
r = np.linspace(r_min, r_max, nr + 1)
t = np.linspace(0, t_max, nt + 1)

T_forward = np.zeros((nr + 1, nt + 1))
T_backward = np.zeros_like(T_forward)
T_cn = np.zeros_like(T_forward)

T_forward[:, 0] = 200 * (r - 0.5)
T_backward[:, 0] = 200 * (r - 0.5)
T_cn[:, 0] = 200 * (r - 0.5)

T_forward[-1, :] = 100 + 40 * t
T_backward[-1, :] = 100 + 40 * t
T_cn[-1, :] = 100 + 40 * t

def apply_neumann(T, t_idx):
    if T.ndim == 1:
        T[0] = T[1] / (1 + 3 * dr)
    elif T.ndim == 2:
        T[0, t_idx] = T[1, t_idx] / (1 + 3 * dr)

for n in range(0, nt):
    for i in range(1, nr):
        T_forward[i, n+1] = T_forward[i, n] + alpha * (
            T_forward[i+1, n] - 2*T_forward[i, n] + T_forward[i-1, n] +
            dr / r[i] * (T_forward[i+1, n] - T_forward[i-1, n]) / 2
        )
    apply_neumann(T_forward, n+1)

for n in range(0, nt):
    T_new = T_backward[:, n].copy()
    for it in range(100):
        T_old = T_new.copy()
        for i in range(1, nr):
            T_new[i] = (
                T_backward[i, n] + alpha * (
                    T_old[i+1] + T_old[i-1] +
                    dr / r[i] * (T_old[i+1] - T_old[i-1]) / 2
                )
            ) / (1 + 2 * alpha)
        apply_neumann(T_new, 0)
        if np.linalg.norm(T_new - T_old) < 1e-4:
            break
    T_backward[:, n+1] = T_new

for n in range(0, nt):
    for i in range(1, nr):
        A = alpha / 2
        B = 1 + alpha
        rhs = T_cn[i, n] + A * (
            T_cn[i+1, n] - 2*T_cn[i, n] + T_cn[i-1, n] +
            dr / r[i] * (T_cn[i+1, n] - T_cn[i-1, n]) / 2
        )
        T_cn[i, n+1] = rhs / B
    apply_neumann(T_cn, n+1)

plt.figure(figsize=(10, 6))
plt.plot(r, T_forward[:, -1], label='Forward Difference')
plt.plot(r, T_backward[:, -1], label='Backward Difference')
plt.plot(r, T_cn[:, -1], label='Crank-Nicolson')
plt.xlabel('r')
plt.ylabel('T(r, t=10)')
plt.title('Problem 2: Temperature Distribution at t=10')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
