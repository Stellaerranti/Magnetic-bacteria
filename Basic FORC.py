import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Define magnetic field range
H = np.linspace(-1, 1, 100)  # Magnetic field

# Generate synthetic hysteresis data
M = np.zeros((len(H), len(H)))  # Magnetization array

for i, Ha in enumerate(H):
    for j, Hb in enumerate(H):
        if Hb >= Ha:
            M[i, j] = 1 - 2 * np.exp(-10 * (Hb - Ha))
        else:
            M[i, j] = -1 + 2 * np.exp(-10 * (Ha - Hb))

# Calculate FORC function using second derivatives
FORC = np.zeros_like(M)
for i in range(1, len(H) - 1):
    for j in range(1, len(H) - 1):
        d2M = (M[i + 1, j + 1] - M[i + 1, j - 1] - M[i - 1, j + 1] + M[i - 1, j - 1]) / 4
        FORC[i, j] = d2M

# Apply Gaussian smoothing
FORC_smoothed = gaussian_filter(FORC, sigma=1)

# Convert to Hc and Hu coordinates
Hc = (H[None, :] - H[:, None]) / 2
Hu = (H[None, :] + H[:, None]) / 2

# Plot the FORC diagram
plt.figure(figsize=(8, 6))
plt.contourf(Hc, Hu, FORC_smoothed, levels=100, cmap='RdBu_r')
plt.colorbar(label='FORC Intensity')
plt.xlabel('Hc (Coercivity)')
plt.ylabel('Hu (Interaction Field)')
plt.title('FORC Diagram')
plt.grid()
plt.show()
