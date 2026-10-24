import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from heightmap import MAX_HEIGHT, generate_heightmap


def show_heightmap(heightmap: np.ndarray) -> None:
    x, y, z = heightmap.shape[0], heightmap.shape[1], heightmap.max()

    # Layers of each height
    layers = np.arange(z)[None, None, :]

    # Turn 2d height map into 3d voxel map (True if the cell is non-empty, False if empty)
    voxel_map = layers < heightmap[..., None]

    grid = pv.ImageData(dimensions=(x + 1, y + 1, z + 1))

    # -1 if the current grid is empty at a height layer (ex. if a grid has height 1, then layer1 = 1 and layer2 = -1, etc...)
    grid.cell_data["height"] = np.where(voxel_map, layers, -1).ravel(order="F")

    # Remove all the empty cells
    blocks = grid.threshold(0)
    blocks.plot(show_edges=False, cmap="gray")

if __name__ == "__main__":
    rng = np.random.default_rng(100)
    heightmap = generate_heightmap(16, rng)
    print(heightmap, flush=True)
    show_heightmap(heightmap)
