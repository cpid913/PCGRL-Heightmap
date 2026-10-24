import matplotlib.pyplot as plt
import numpy as np

from heightmap import MAX_HEIGHT, generate_heightmap


def show_heightmap(heightmap: np.ndarray) -> None:
    """Draw each cell as a block column colored by its discrete height."""
    n = heightmap.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    heights = heightmap.ravel()
    colors = plt.cm.terrain(0.35 + 0.65 * (heights - 1) / (MAX_HEIGHT - 1))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xs.ravel(),
        ys.ravel(),
        np.zeros_like(heights),
        1,
        1,
        heights,
        color=colors,
        shade=True,
        edgecolor="black",
        linewidth=0.2,
    )
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, MAX_HEIGHT)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("height")
    ax.set_title("16x16 block heightmap")
    ax.view_init(elev=35, azim=-60)
    plt.tight_layout()
    plt.show()


rng = np.random.default_rng(100)
heightmap = generate_heightmap(16, rng)
print(heightmap, flush=True)
show_heightmap(heightmap)
