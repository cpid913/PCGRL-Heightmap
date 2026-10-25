import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from heightmap import MAX_HEIGHT, generate_heightmap
from path_finding import Astar


def show_heightmap(heightmap, area, start: tuple, goal: tuple) -> None:
    x, y, z = heightmap.shape[0], heightmap.shape[1], heightmap.max()

    # A binary map of whether the current grid is explored
    highlight = np.zeros(heightmap.shape, dtype=bool)
    for i, j in area:
        highlight[i, j] = True

    dots = {start: "yellow", goal: "red"}

    # Layers of each height
    layers = np.arange(z)[None, None, :]

    # Turn 2d height map into 3d voxel map (True if the cell is non-empty, False if empty)
    voxel_map = layers < heightmap[..., None]

    grid = pv.ImageData(dimensions=(x + 1, y + 1, z + 1))

    # -1 if the current grid is empty at a height layer (ex. if a grid has height 1, then layer1 = 1 and layer2 = -1, etc...)
    grid.cell_data["height"] = np.where(voxel_map, layers, -1).ravel(order="F")

    # Replace the top voxel of a grid to a highlighted voxel if the grid is visited
    top = layers == (heightmap[..., None] - 1)
    grid.cell_data["highlight"] = (top & highlight[..., None]).astype(int).ravel(order="F")

    # Remove all the empty cells
    blocks = grid.threshold(0, scalars="height")

    # Split the voxels into explored and unexplored
    normal = blocks.threshold(0.5, scalars="highlight", invert=True)  # highlight == 0
    marked = blocks.threshold(0.5, scalars="highlight")

    p = pv.Plotter()

    # Plot unexplored voxels
    p.add_mesh(normal, scalars="height", cmap="gray", show_scalar_bar=False)
    # Plot highlighted voxels
    p.add_mesh(marked, color="light_coral")

    # Plot spheres representing the start point and the main quest point
    pts = np.array([[i + 0.5, j + 0.5, heightmap[i, j] + 0.5] for i, j in dots], dtype=float)
    cols = np.array([pv.Color(c).int_rgb for c in dots.values()], dtype=np.uint8)
    pv_points = pv.PolyData(pts)
    pv_points["colors"] = cols
    spheres = pv_points.glyph(geom=pv.Sphere(radius=0.5), scale=False, orient=False)
    p.add_mesh(spheres, scalars="colors", rgb=True)
    p.show()

if __name__ == "__main__":
    rng = np.random.default_rng(100)
    heightmap = generate_heightmap(16, rng)

    start, goal = (6, 2), (8, 15)
    path, area = Astar(heightmap, start, goal)

    if path:
        print(f"Path found ({len(path) - 1} steps)\n")
    else:
        print("No path found.")

    print(heightmap, flush=True)
    show_heightmap(heightmap, area, start, goal)
