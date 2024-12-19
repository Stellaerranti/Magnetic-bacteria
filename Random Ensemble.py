import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Parameters for the particle ensemble
Ms = 1.0  # Saturation magnetization
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability

# Generate particle parameters
np.random.seed(42)  # For reproducibility
num_particles = 1000
anisotropy_fields = np.random.uniform(0.8, 1.2, num_particles)  # Random HK values
orientations = np.random.uniform(0, np.pi, num_particles)  # Random orientations

# Define the energy function
def energy(theta, H, phi, HK):
    return -HK * np.cos(2 * (theta - phi)) - Ms * H * np.cos(theta)

# Solve for equilibrium angle of magnetization
def equilibrium_angle(H, phi, HK):
    theta_vals = np.linspace(0, 2 * np.pi, 1000)
    energies = energy(theta_vals, H, phi, HK)
    min_idx = np.argmin(energies)
    return theta_vals[min_idx]

# Generate magnetization data for the particle ensemble
def generate_magnetization(H_range, anisotropy_fields, orientations):
    M = np.zeros((len(H_range), len(H_range)))
    for i, Ha in enumerate(tqdm(H_range, desc="Generating FORCs")):
        for j, Hb in enumerate(H_range):
            if Hb >= Ha:
                magnetization = 0
                for k in range(len(anisotropy_fields)):
                    HK = anisotropy_fields[k]
                    phi = orientations[k]
                    theta = equilibrium_angle(Hb, phi, HK)
                    magnetization += Ms * np.cos(theta)
                M[i, j] = magnetization / len(anisotropy_fields)
            else:
                M[i, j] = 0
    return M

# Calculate the FORC distribution
def calculate_forc_distribution(M, H_range):
    FORC = np.zeros_like(M)
    for i in range(1, len(H_range) - 1):
        for j in range(1, len(H_range) - 1):
            d2M = (M[i + 1, j + 1] - M[i + 1, j - 1] - M[i - 1, j + 1] + M[i - 1, j - 1]) / 4
            FORC[i, j] = d2M
    return FORC

# Normalize data for plotting
def normalize_data(data):
    return data / np.max(np.abs(data))

# Main parameters
H_range = np.linspace(-2.0, 2.0, 300)  # Magnetic field range

# Generate FORC magnetization data
M = generate_magnetization(H_range, anisotropy_fields, orientations)

# Calculate FORC distribution
FORC = calculate_forc_distribution(M, H_range)

# Normalize FORC data
FORC_normalized = normalize_data(FORC)

# Apply Gaussian smoothing
FORC_smoothed = gaussian_filter(FORC_normalized, sigma=1.0)

# Convert to Hc and Hu coordinates
Hc = (H_range[None, :] - H_range[:, None]) / 2
Hu = (H_range[None, :] + H_range[:, None]) / 2

# Plot the FORC diagram
plt.figure(figsize=(8, 6))
plt.contourf(Hc, Hu, FORC_smoothed, levels=100, cmap='RdBu_r')
plt.colorbar(label='Normalized FORC Intensity')
plt.xlabel('Hc (Coercivity)')
plt.ylabel('Hu (Interaction Field)')
plt.title('FORC Diagram for Particle Ensemble with Variations')
plt.grid()
plt.show()
