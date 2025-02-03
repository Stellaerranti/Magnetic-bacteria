import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Constants for LLG Model
mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
gamma = 2.21e5  # Gyromagnetic ratio (m/A·s)
alpha = 0.1  # Damping constant
K = 1e4  # Anisotropy constant (J/m^3)
Ms = 1e6  # Saturation magnetization (A/m)
N = 10  # Number of particles

# Generate Random Positions and Easy Axes
positions = np.random.rand(N, 3) * 1e-6  # Random positions in 3D space (m)
easy_axes = np.random.rand(N, 3)
easy_axes /= np.linalg.norm(easy_axes, axis=1, keepdims=True)  # Normalize

# Define External Field Range
H_max = 50e3  # Maximum applied field (A/m)
H_step = 5e3  # Field step (A/m)
H_values = np.arange(-H_max, H_max + H_step, H_step)

# Compute Dipole-Dipole Interactions
def compute_dipole_fields(m):
    r_ij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1, keepdims=True)
    np.fill_diagonal(r_norm[:, :, 0], np.inf)
    r_unit = np.where(r_norm > 0, r_ij / r_norm, 0)
    m_dot_r = np.sum(m[np.newaxis, :, :] * r_unit, axis=-1, keepdims=True)
    term1 = 3 * m_dot_r * r_unit / r_norm**5
    term2 = m[np.newaxis, :, :] / r_norm**3
    H_dipole = np.sum((term1 - term2) / (4 * np.pi), axis=1)
    return H_dipole

# LLG Solver
def llg_equation(t, m_flat, H_ext):
    m = m_flat.reshape(N, 3)
    H_aniso = (2 * K / (mu0 * Ms)) * np.sum(m * easy_axes, axis=1, keepdims=True) * easy_axes
    H_dipole = compute_dipole_fields(m)
    H_eff = H_ext + H_aniso + H_dipole
    precession = -gamma * np.cross(m, H_eff)
    damping = alpha * np.cross(m, precession)
    return (precession + damping).flatten()

# FORC Protocol with LLG Integration
M_values = np.zeros_like(H_values, dtype=np.float64)
for idx, H_ext_mag in tqdm(enumerate(H_values), total=len(H_values), desc='Computing FORCs'):
    H_ext = np.array([H_ext_mag, 0, 0])
    m0 = easy_axes.copy()
    m0_flat = m0.flatten()
    sol = solve_ivp(llg_equation, [0, 1e-9], m0_flat, args=(H_ext,), method="RK45", t_eval=[1e-9])
    m_final = sol.y[:, -1].reshape(N, 3)
    M_values[idx] = np.sum(m_final[:, 0])

# Normalize and Compute FORC Distribution
M_values /= N
FORC = np.zeros((len(H_values), len(H_values)))
for i, Hr in tqdm(enumerate(H_values), total=len(H_values), desc='Computing FORC Diagram'):
    for j, Hb in enumerate(H_values):
        if Hb > Hr:
            idx_Hb = np.where(H_values == Hb)[0][0]
            idx_Hr = np.where(H_values == Hr)[0][0]
            FORC[i, j] = (M_values[idx_Hb] - M_values[idx_Hr]) / (Hb - Hr)

# Plot FORC Diagram
plt.figure(figsize=(8, 6))
plt.imshow(FORC, extent=[-H_max, H_max, -H_max, H_max], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='FORC Distribution')
plt.xlabel('Hb (Applied Field)')
plt.ylabel('Hr (Reversal Field)')
plt.title('FORC Diagram (sLLG Model)')
plt.show()