import numpy as np
from scipy.special import expit

# Close to Minecraft's octave stack, sized for a 16x16 block map.
# Four layers: each one is twice as fine and half as strong as the last.
# A fifth octave starts flickering between neighboring blocks.
OCTAVES = 4
PERSISTENCE = 0.5
LACUNARITY = 2.0
BASE_FREQUENCY = 1.0
CONTRAST = 1.0
MAX_HEIGHT = 8  # column heights are the integers 1..MAX_HEIGHT


def _perlin(map_size: int, rng: np.random.Generator, frequency: float) -> np.ndarray:
    """One octave of gradient noise. frequency is how many cycles fit across the map."""
    # Sample at cell centers so a block isn't sitting on a lattice point (those are 0).
    noise_coord = (np.arange(map_size) + 0.5) * (frequency / map_size)
    lattice_cell = np.floor(noise_coord).astype(np.int32)
    offset_in_cell = noise_coord - lattice_cell

    # +3 covers the far corner of the last cell, with a spare.
    lattice_size = int(lattice_cell.max()) + 3
    gradient_angle = rng.uniform(0.0, 2.0 * np.pi, size=(lattice_size, lattice_size))
    gradient_row = np.cos(gradient_angle)
    gradient_col = np.sin(gradient_angle)

    # Smoothstep: flat at the cell edges, steeper in the middle.
    fade = offset_in_cell * offset_in_cell * (3.0 - 2.0 * offset_in_cell)

    row, col = np.meshgrid(lattice_cell, lattice_cell, indexing="ij")
    offset_row, offset_col = np.meshgrid(offset_in_cell, offset_in_cell, indexing="ij")
    fade_row, fade_col = np.meshgrid(fade, fade, indexing="ij")

    def gradient_dot(corner_row, corner_col, dist_row, dist_col):
        return (
            gradient_row[corner_row, corner_col] * dist_row
            + gradient_col[corner_row, corner_col] * dist_col
        )

    # Four corners of the cell. The distance is measured from that corner to the sample.
    corner_row0_col0 = gradient_dot(row, col, offset_row, offset_col)
    corner_row1_col0 = gradient_dot(row + 1, col, offset_row - 1.0, offset_col)
    corner_row0_col1 = gradient_dot(row, col + 1, offset_row, offset_col - 1.0)
    corner_row1_col1 = gradient_dot(row + 1, col + 1, offset_row - 1.0, offset_col - 1.0)

    at_col0 = corner_row0_col0 * (1.0 - fade_row) + corner_row1_col0 * fade_row
    at_col1 = corner_row0_col1 * (1.0 - fade_row) + corner_row1_col1 * fade_row
    return at_col0 * (1.0 - fade_col) + at_col1 * fade_col


def _normalize(values: np.ndarray) -> np.ndarray:
    lowest = float(values.min())
    highest = float(values.max())
    if highest - lowest < 1e-12:
        return np.full(values.shape, 0.5)
    return (values - lowest) / (highest - lowest)


def generate_heightmap(map_size: int, rng: np.random.Generator) -> np.ndarray:
    """map_size by map_size int32 heights in 1..MAX_HEIGHT. Same rng state, same map."""
    if map_size < 2:
        raise ValueError(f"map size must be at least 2, got {map_size}")

    height_sum = np.zeros((map_size, map_size))
    amplitude = 1.0
    frequency = BASE_FREQUENCY
    amplitude_sum = 0.0

    for _ in range(OCTAVES):
        height_sum += amplitude * _perlin(map_size, rng, frequency)
        amplitude_sum += amplitude
        amplitude *= PERSISTENCE
        frequency *= LACUNARITY

    normalized = _normalize(height_sum / amplitude_sum)
    # expit flattens the extremes but never quite reaches 0 or 1.
    shaped = _normalize(expit(CONTRAST * (normalized - 0.5)))
    return np.rint(shaped * (MAX_HEIGHT - 1)).astype(np.int32) + 1
