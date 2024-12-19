import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  # For progress bar

def generate_hysteresis_loops_with_chains(num_points=300, num_loops=150, field_max=1.0, num_chains=10, chain_size=100):
    """
    Generate synthetic hysteresis loops for chains of single-domain particles.

    Parameters:
        num_points (int): Number of field points in each loop.
        num_loops (int): Number of reversal loops.
        field_max (float): Maximum applied field.
        num_chains (int): Number of chains of particles.
        chain_size (int): Number of particles in each chain.

    Returns:
        Ha, Hb, M (numpy arrays): Arrays of reversal fields, applied fields, and magnetizations.
    """
    fields = np.linspace(-field_max, field_max, num_points)
    Ha = np.linspace(-field_max, field_max, num_loops)
    Hb = fields
    M = np.zeros((num_loops, num_points))

    # Generate chains of particles
    for chain in tqdm(range(num_chains), desc="Generating chains", leave=True):
        # Randomize chain properties
        chain_coercivities = np.random.normal(0.5 * field_max, 0.1 * field_max, chain_size)
        chain_offsets = np.random.normal(0, 0.2 * field_max, chain_size)

        for particle in range(chain_size):
            for i, h_a in enumerate(Ha):
                for j, h_b in enumerate(fields):
                    if h_b >= h_a + chain_offsets[particle]:
                        # Gaussian-like switching for magnetization with chain interactions
                        M[i, j] += np.exp(-((h_b - chain_offsets[particle])**2) / (2 * chain_coercivities[particle]**2))

    # Normalize by the total number of particles
    total_particles = num_chains * chain_size
    M /= total_particles

    return Ha, Hb, M

def calculate_forc_distribution(Ha, Hb, M, smoothing_sigma=2):
    """
    Calculate the FORC distribution from hysteresis loops.

    Parameters:
        Ha (numpy array): Reversal fields.
        Hb (numpy array): Applied fields.
        M (numpy array): Magnetization values.
        smoothing_sigma (int): Standard deviation for Gaussian smoothing.

    Returns:
        H_c, H_u, rho (numpy arrays): Arrays of coercivity, interaction field, and normalized FORC distribution.
    """
    num_loops, num_points = M.shape
    H_c = np.zeros_like(M)
    H_u = np.zeros_like(M)
    rho = np.zeros_like(M)

    for i in tqdm(range(1, num_loops), desc="Calculating FORC", leave=True):
        for j in range(1, num_points):
            # Calculate mixed second derivative
            dM_dHa = (M[i, j] - M[i - 1, j]) / (Ha[i] - Ha[i - 1])
            dM_dHb = (M[i, j] - M[i, j - 1]) / (Hb[j] - Hb[j - 1])
            rho[i, j] = (dM_dHb - dM_dHa) / ((Hb[j] - Hb[j - 1]) * (Ha[i] - Ha[i - 1]))

    # Apply Gaussian smoothing
    rho = gaussian_filter(rho, sigma=smoothing_sigma)

    # Normalize the FORC distribution to [-1, 1]
    rho /= np.nanmax(np.abs(rho))

    # Convert to H_c and H_u coordinates
    H_c = (Hb[np.newaxis, :] - Ha[:, np.newaxis]) / 2
    H_u = (Hb[np.newaxis, :] + Ha[:, np.newaxis]) / 2

    return H_c, H_u, rho

# Generate synthetic data for chains of particles
Ha, Hb, M = generate_hysteresis_loops_with_chains(num_points=300, num_loops=150, field_max=1.0, num_chains=10, chain_size=100)

# Calculate FORC distribution
H_c, H_u, rho = calculate_forc_distribution(Ha, Hb, M, smoothing_sigma=3)

# Plot FORC diagram
plt.figure(figsize=(8, 6))
contour = plt.contourf(H_u, H_c, rho, levels=np.linspace(-1, 1, 100), cmap="viridis", extend="both")
plt.colorbar(contour, label="Normalized FORC Distribution (rho)")
plt.contour(H_u, H_c, rho, levels=np.linspace(-1, 1, 10), colors='black', linewidths=0.5, alpha=0.5)
plt.title("FORC Diagram for Chains of Particles")
plt.xlabel("$H_u$ (Interaction Field)")
plt.ylabel("$H_c$ (Coercivity)")
plt.show()
