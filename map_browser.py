import json
import os
import sys

import numpy as np
import pyvista as pv

from path_finding import floodfill

PAGE = 12

'''
<folder>/1.json, <folder>/2.json, ... in number order. The folder defaults to maps.
Floodfill is run again so the coral tops match the saved quests.
Maps written by evaluate.py also carry original_heightmap, the terrain before the agent edited it.
'''
def load_maps(folder_name="maps"):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)
    names = [name for name in os.listdir(folder) if name.endswith(".json") and name[:-5].isdigit()]
    names.sort(key=lambda name: int(name[:-5]))

    batch = []
    for name in names:
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            raw = json.load(f)

        item = {
            "id": raw["id"],
            "seed": raw["seed"],
            "spawn": tuple(raw["spawn"]),
            "main_quest": tuple(raw["main_quest"]),
            "side_quests": [tuple(cell) for cell in raw["side_quests"]],
            "demand": raw["requested_reachable"],
            "edited_heightmap": np.array(raw["heightmap"], dtype=np.int32),
            "original_heightmap": None,
        }
        if "original_heightmap" in raw:
            item["original_heightmap"] = np.array(raw["original_heightmap"], dtype=np.int32)
        batch.append(item)
    return batch


'''
The item drawn on screen: the chosen heightmap with its floodfill and reached quests worked out.
'''
def shown_version(item, show_original):
    use_original = show_original and item["original_heightmap"] is not None
    heightmap = item["original_heightmap"] if use_original else item["edited_heightmap"]
    visited = floodfill(heightmap, item["spawn"])
    shown = dict(item)
    shown["heightmap"] = heightmap
    shown["visited"] = visited
    shown["reached_sides"] = [cell for cell in item["side_quests"] if cell in visited]
    shown["main_reached"] = item["main_quest"] in visited
    if item["original_heightmap"] is not None:
        shown["version"] = "original" if use_original else "edited"
    return shown


# Marker pins are circles. The two block colors are squares.
LEGEND = [
    ["spawn", "magenta", "circle"],
    ["main quest", "red", "circle"],
    ["side quest reached", "dodgerblue", "circle"],
    ["side quest missed", "darkorange", "circle"],
    ["terrain height", "gray", "rectangle"],
    ["floodfill top", "lightcoral", "rectangle"],
]


def add_color_legend(plotter, size):
    plotter.add_legend(
        LEGEND,
        bcolor="white",
        border=True,
        size=size,
        loc="lower right",
        background_opacity=0.94,
    )


def blurb(item):
    placed = len(item["side_quests"])
    got = len(item["reached_sides"])
    # Too many reachable side quests misses the demand just like too few
    met = "met" if got == item["demand"] else "missed"
    main_txt = "main ok" if item["main_reached"] else "main blocked"
    version = f"  [{item['version']}]" if "version" in item else ""
    return (
        f"map {item['id']}  seed {item['seed']}  demand {item['demand']} ({met}){version}\n"
        f"sides {placed} placed, {got} reached, {main_txt}"
    )


'''
Same voxel columns as main.py. Unvisited blocks are shaded by height (gray),
and the floodfill recolors the top block.
'''
def add_heightmap(plotter, item, font_size):
    heightmap = item["heightmap"]
    x, y, z = heightmap.shape[0], heightmap.shape[1], int(heightmap.max())

    highlight = np.zeros(heightmap.shape, dtype=bool)
    for i, j in item["visited"]:
        highlight[i, j] = True

    layers = np.arange(z)[None, None, :]
    voxel_map = layers < heightmap[..., None]

    grid = pv.ImageData(dimensions=(x + 1, y + 1, z + 1))
    grid.cell_data["height"] = np.where(voxel_map, layers, -1).ravel(order="F")
    top = layers == (heightmap[..., None] - 1)
    grid.cell_data["highlight"] = (top & highlight[..., None]).astype(int).ravel(order="F")

    blocks = grid.threshold(0, scalars="height")
    normal = blocks.threshold(0.5, scalars="highlight", invert=True)
    marked = blocks.threshold(0.5, scalars="highlight")

    plotter.set_background("white")
    if normal.n_cells:
        plotter.add_mesh(normal, scalars="height", cmap="gray", show_scalar_bar=False)
    if marked.n_cells:
        plotter.add_mesh(marked, color="light_coral")

    reached = set(item["reached_sides"])
    dots = [(item["spawn"], "magenta"), (item["main_quest"], "red")]
    for cell in item["side_quests"]:
        color = "dodgerblue" if cell in reached else "darkorange"
        dots.append((cell, color))

    # Stem up to one shared height so a marker in a hole is not hidden by a taller column
    roof = float(heightmap.max()) + 1.6
    for (i, j), color in dots:
        plotter.add_mesh(
            pv.Line((i + 0.5, j + 0.5, float(heightmap[i, j])), (i + 0.5, j + 0.5, roof)),
            color=color,
            line_width=3,
        )

    pts = np.array([[i + 0.5, j + 0.5, roof] for (i, j), _c in dots], dtype=float)
    cols = np.array([pv.Color(c).int_rgb for _cell, c in dots], dtype=np.uint8)
    cloud = pv.PolyData(pts)
    cloud["colors"] = cols
    spheres = cloud.glyph(geom=pv.Sphere(radius=0.42, theta_resolution=12, phi_resolution=12), scale=False, orient=False)
    plotter.add_mesh(spheres, scalars="colors", rgb=True, smooth_shading=True)

    plotter.add_text(blurb(item), font_size=font_size, color="black")


'''
Three-quarter view, up is height, so a tall column reads as tall.
'''
def aim_camera(plotter, heightmap):
    rows, cols = heightmap.shape
    # Far enough back that the pins above the tallest column stay inside the frame
    roof = float(heightmap.max()) + 2.2
    center = (rows / 2.0, cols / 2.0, roof * 0.42)
    dist = max(rows, cols) * 2.2
    plotter.camera_position = [
        (center[0] + dist * 0.72, center[1] - dist, center[2] + dist * 0.7),
        center,
        (0, 0, 1),
    ]


class Browser:
    def __init__(self, folder_name="maps"):
        self.index = 0
        self.together = False
        self.switching = False
        self.show_original = False
        self.plotter = None
        self.folder_name = folder_name
        self.batch = load_maps(folder_name)
        print(f"Loaded {len(self.batch)} maps from {folder_name}/")

    def make_plotter(self):
        if self.together:
            # 3x4 maps, plus a right-hand column for the color legend
            plotter = pv.Plotter(shape=(3, 5), window_size=(1680, 980), title="Heightmap batch")
        else:
            plotter = pv.Plotter(window_size=(1000, 820), title="Heightmap batch")
        plotter.add_key_event("Left", self.on_prev)
        plotter.add_key_event("Right", self.on_next)
        plotter.add_key_event("a", self.on_mode)
        plotter.add_key_event("n", self.on_new)
        plotter.add_key_event("b", self.on_before_after)
        return plotter

    def draw(self):
        plotter = self.plotter
        plotter.clear()
        if self.together:
            page_start = (self.index // PAGE) * PAGE
            page = self.batch[page_start:page_start + PAGE]
            for slot in range(PAGE):
                plotter.subplot(slot // 4, slot % 4)
                plotter.set_background("white")
                if slot < len(page):
                    item = shown_version(page[slot], self.show_original)
                    add_heightmap(plotter, item, font_size=8)
                    aim_camera(plotter, item["heightmap"])
            for row in range(3):
                plotter.subplot(row, 4)
                plotter.set_background("white")
            plotter.subplot(1, 4)
            add_color_legend(plotter, size=(0.92, 0.86))
            plotter.subplot(0, 4)
            plotter.add_text(
                "left/right  one map\na  back to one\nn  reload maps\nb  before/after edits",
                position="upper_left",
                font_size=12,
                color="black",
            )
            plotter.link_views()
        else:
            item = shown_version(self.batch[self.index], self.show_original)
            add_heightmap(plotter, item, font_size=13)
            aim_camera(plotter, item["heightmap"])
            add_color_legend(plotter, size=(0.24, 0.28))
            plotter.add_text(
                f"map {self.index + 1}/{len(self.batch)}    "
                "left/right  next map    a  page of 12    n  reload    b  before/after",
                position="lower_left",
                font_size=11,
                color="black",
            )
        plotter.render()

    def on_prev(self):
        self.index = (self.index - 1) % len(self.batch)
        if self.together:
            self.on_mode()
            return
        self.draw()

    def on_next(self):
        self.index = (self.index + 1) % len(self.batch)
        if self.together:
            self.on_mode()
            return
        self.draw()

    def on_mode(self):
        self.together = not self.together
        self.switching = True
        self.plotter.close()

    def on_new(self):
        self.batch = load_maps(self.folder_name)
        print(f"Loaded {len(self.batch)} maps from {self.folder_name}/")
        if not self.batch:
            return
        self.index = min(self.index, len(self.batch) - 1)
        self.draw()

    # Only maps written by evaluate.py have an original to switch to
    def on_before_after(self):
        self.show_original = not self.show_original
        self.draw()

    def run(self):
        while True:
            self.switching = False
            self.plotter = self.make_plotter()
            self.draw()
            self.plotter.show()
            if not self.switching:
                break


if __name__ == "__main__":
    # python map_browser.py              the generated maps
    # python map_browser.py maps_edited  the agent's edits, b switches to the original
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "maps"
    browser = Browser(folder_name)
    if browser.batch:
        browser.run()
    else:
        print(f"No maps in the {folder_name} folder. Run bulk_generate.py or evaluate.py first.")
