'''
Written by Claude Code:
Before/after view of a trained PQN policy on the run's held-out maps, side by side in one window.
Left: the map as it started. Right: the map after the greedy policy edited it.
Left/Right arrow keys switch maps. Drawing reuses map_browser.py.

Usage:
  python view_policy.py runs/<run_name>                   # uses best_model.pt
  python view_policy.py runs/<run_name> model.pt          # any model file in the run folder
'''
import os
import sys
from types import SimpleNamespace

import pyvista as pv
import torch

from env import PQNEnv
from evaluate_pqn import evaluate, load_eval_maps
from map_browser import add_color_legend, add_heightmap, aim_camera
from path_finding import floodfill_stop
from pqn import QNetwork


def shown(m, heightmap, label):
    '''The item add_heightmap draws. Uses floodfill_stop, like the env, so the counts match evaluation.'''
    visited = floodfill_stop(heightmap, m["spawn"], m["main_quest"])
    item = dict(m)
    item["heightmap"] = heightmap
    item["visited"] = visited
    item["reached_sides"] = [cell for cell in m["side_quests"] if cell in visited]
    item["main_reached"] = m["main_quest"] in visited
    item["version"] = label
    return item


class PolicyViewer:
    def __init__(self, maps, results):
        self.maps = maps
        self.results = results
        self.index = 0
        self.plotter = pv.Plotter(shape=(1, 2), window_size=(1700, 850), title="Policy before/after")
        self.plotter.add_key_event("Left", lambda: self.step(-1))
        self.plotter.add_key_event("Right", lambda: self.step(1))

    def step(self, delta):
        self.index = (self.index + delta) % len(self.maps)
        self.draw()

    def draw(self):
        m, r = self.maps[self.index], self.results[self.index]
        before = shown(m, m["heightmap"], "before")
        after = shown(m, r["final_heightmap"], "after")
        self.plotter.clear()
        for col, item in enumerate([before, after]):
            self.plotter.subplot(0, col)
            add_heightmap(self.plotter, item, font_size=12)
            aim_camera(self.plotter, item["heightmap"])
        add_color_legend(self.plotter, size=(0.28, 0.3))
        self.plotter.add_text(
            f"map {self.index + 1}/{len(self.maps)}   solved={r['terminated']}   steps={r['length']}   "
            f"cells changed={r['cells_changed']}      left/right  next map",
            position="lower_left", font_size=11, color="black",
        )
        self.plotter.link_views()
        self.plotter.render()


def main():
    if len(sys.argv) < 2:
        print("usage: python view_policy.py runs/<run_name> [model file, default best_model.pt]")
        return
    run_dir = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "best_model.pt"

    maps = load_eval_maps(SimpleNamespace(run_dir=run_dir, seed=1, eval_fraction=0.2))
    probe = PQNEnv(maps)
    q_network = QNetwork(SimpleNamespace(single_observation_space=probe.observation_space,
                                         single_action_space=probe.action_space))
    q_network.load_state_dict(torch.load(os.path.join(run_dir, model_file), map_location="cpu"))

    results = evaluate(q_network, maps, torch.device("cpu"))
    solved = sum(r["terminated"] for r in results)
    print(f"{model_file}: solved {solved}/{len(results)} held-out maps")

    viewer = PolicyViewer(maps, results)
    viewer.draw()
    viewer.plotter.show()


if __name__ == "__main__":
    main()
