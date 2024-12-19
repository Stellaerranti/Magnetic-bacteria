import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Parameters for Stoner-Wohlfarth model
Ku = 1.0  # Uniaxial anisotropy constant
Ms = 1.0  # Saturation magnetization
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
HK = 2 * Ku / (mu0 * Ms)  # Anisotropy field

# Stoner-Wohlfarth energy equation
def energy(theta, h, phi):
    return Ku * np.sin(theta - phi)**2 - mu0 * Ms * h * np.cos(theta)

# Solve for equilibrium magnetization states
def solve_equilibrium(h, phi):
    theta_vals = np.linspace(0, 2 * np.pi, 1000)
    energies = energy(theta_vals, h, phi)
    min_idx = np.argmin(energies)
    return theta_vals[min_idx]

# Generate magnetization for an ensemble of particles
def generate_ensemble(H, phi_distribution):
    M = np.zeros((len(H), len(H)))
    for Ha_idx, Ha in enumerate(tqdm(H, desc="Processing Ensemble")):
        for Hb_idx, Hb in enumerate(H):
            if Hb >= Ha:
                # Iterate over particles with varying phi values
                magnetization = 0
                for phi in phi_distribution:
                    theta = solve_equilibrium(Hb / HK, phi)
                    magnetization += Ms * np.cos(theta)
                M[Ha_idx, Hb_idx] = magnetization / len(phi_distribution)
            else:
                M[Ha_idx, Hb_idx] = 0
    return M

# Calculate FORC distribution using second derivatives
def calculate_forc(M, H):
    FORC = np.zeros_like(M)
    for i in range(1, len(H) - 1):
        for j in range(1, len(H) - 1):
            d2M = (M[i + 1, j + 1] - M[i + 1, j - 1] - M[i - 1, j + 1] + M[i - 1, j - 1]) / 4
            FORC[i, j] = d2M
    return FORC

# Main parameters
H = np.linspace(-2.0 * HK, 2.0 * HK, 200)  # Magnetic field range
phi_distribution = np.linspace(0, np.pi / 2, 50)  # Randomly distributed easy axes

# Generate FORCs for the ensemble
M_ensemble = generate_ensemble(H, phi_distribution)

# Calculate FORC distribution
FORC = calculate_forc(M_ensemble, H)

# Enhance peaks by smoothing
FORC_smoothed = gaussian_filter(FORC, sigma=0.5)

# Convert to Hc and Hu coordinates
Hc = (H[None, :] - H[:, None]) / 2
Hu = (H[None, :] + H[:, None]) / 2

# Plot the FORC diagram
plt.figure(figsize=(8, 6))
plt.contourf(Hc, Hu, FORC_smoothed, levels=100, cmap='RdBu_r')
plt.colorbar(label='FORC Intensity')
plt.xlabel('Hc (Coercivity)')
plt.ylabel('Hu (Interaction Field)')
plt.title('FORC Diagram for Ensemble of Particles')
plt.grid()
plt.show()
 