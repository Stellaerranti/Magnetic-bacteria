import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  # For progress bar

def generate_hysteresis_loops(num_points=300, num_loops=150, field_max=1.0, num_particles=1000):
    """
    Generate synthetic hysteresis loops for an ensemble of single-domain particles.

    Parameters:
        num_points (int): Number of field points in each loop.
        num_loops (int): Number of reversal loops.
        field_max (float): Maximum applied field.
        num_particles (int): Number of particles in the ensemble.

    Returns:
        Ha, Hb, M (numpy arrays): Arrays of reversal fields, applied fields, and magnetizations.
    """
    fields = np.linspace(-field_max, field_max, num_points)
    Ha = np.linspace(-field_max, field_max, num_loops)
    Hb = fields
    M = np.zeros((num_loops, num_points))

    # Adjust particle count to avoid fractional division issues
    group_size = num_particles // 3
    coercivities = np.concatenate([
        np.random.normal(0.3 * field_max, 0.02 * field_max, group_size),
        np.random.normal(0.6 * field_max, 0.03 * field_max, group_size),
        np.random.normal(0.9 * field_max, 0.04 * field_max, num_particles - 2 * group_size)
    ])
    offsets = np.concatenate([
        np.random.normal(-0.4 * field_max, 0.05 * field_max, group_size),
        np.random.normal(0, 0.05 * field_max, group_size),
        np.random.normal(0.4 * field_max, 0.05 * field_max, num_particles - 2 * group_size)
    ])

    for p in tqdm(range(num_particles), desc="Generating hysteresis loops", leave=True):
        for i, h_a in enumerate(Ha):
            for j, h_b in enumerate(fields):
                if h_b >= h_a + offsets[p]:
                    # Gaussian-like switching for magnetization with sharp peaks
                    M[i, j] += np.exp(-((h_b - offsets[p])**2) / (2 * coercivities[p]**2))

    # Normalize by the number of particles
    M /= num_particles

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

# Generate synthetic data for an ensemble of particles
Ha, Hb, M = generate_hysteresis_loops(num_points=300, num_loops=150, field_max=1.0, num_particles=1000)

# Calculate FORC distribution
H_c, H_u, rho = calculate_forc_distribution(Ha, Hb, M, smoothing_sigma=3)

# Plot FORC diagram
plt.figure(figsize=(8, 6))
contour = plt.contourf(H_u, H_c, rho, levels=np.linspace(-1, 1, 100), cmap="viridis", extend="both")
plt.colorbar(contour, label="Normalized FORC Distribution (rho)")
plt.contour(H_u, H_c, rho, levels=np.linspace(-1, 1, 10), colors='black', linewidths=0.5, alpha=0.5)
plt.title("FORC Diagram")
plt.xlabel("$H_u$ (Interaction Field)")
plt.ylabel("$H_c$ (Coercivity)")
plt.show()
