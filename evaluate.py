import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

from heightmap_env import HeightmapEnv, load_split
from ppo_train import Agent, play_map_greedily

'''
Plays every test map with a trained agent and writes the edited maps to maps_edited/.
Usage: python evaluate.py runs/<run_name>/agent.pt [maps folder]
'''
def main():
    if len(sys.argv) < 2:
        print("usage: python evaluate.py runs/<run_name>/agent.pt [maps folder]")
        return
    checkpoint_path = sys.argv[1]
    maps_folder = sys.argv[2] if len(sys.argv) > 2 else "maps"
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps_edited")
    os.makedirs(output_folder, exist_ok=True)
    for name in os.listdir(output_folder):
        if name.endswith(".json") and name[:-5].isdigit():
            os.remove(os.path.join(output_folder, name))

    test_records = load_split(maps_folder, "test")
    env = HeightmapEnv(test_records, fixed_order=True)

    # Agent only reads the observation and action sizes from this
    shape_source = gym.vector.SyncVectorEnv([lambda: HeightmapEnv(test_records)])
    agent = Agent(shape_source)
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.eval()

    solved_count = 0
    edits_per_map = []
    side_misses = []
    print(f"{'map':>4} {'start':>5} {'end':>4} {'want':>4} {'main':>5} {'solved':>6} {'edits':>5}")
    for record in test_records:
        final_info = play_map_greedily(agent, env, torch.device("cpu"))
        solved = final_info["success"]
        solved_count += solved
        edits_per_map.append(env.edits_made)
        side_misses.append(abs(env.side_count - env.requested_side_count))

        print(
            f"{record['id']:>4} {env.start_side_count:>5} {env.side_count:>4} {env.requested_side_count:>4} "
            f"{str(env.main_reached):>5} {str(solved):>6} {env.edits_made:>5}"
        )
        save_edited_map(output_folder, record, env)

    print()
    print(f"solved {solved_count}/{len(test_records)} ({solved_count / len(test_records):.0%})")
    print(f"average edits {np.mean(edits_per_map):.1f}")
    print(f"average side quest miss |reached - requested| {np.mean(side_misses):.2f}")
    print(f"edited maps written to {output_folder}")


'''
Same fields as bulk_generate.save_map, so map_browser.py can open the folder.
original_heightmap is the map before the agent touched it.
'''
def save_edited_map(folder, record, env):
    result = {
        "id": record["id"],
        "seed": record["seed"],
        "spawn": list(env.spawn),
        "main_quest": list(env.main_quest),
        "side_quests": [list(cell) for cell in env.side_quests],
        "requested_reachable": env.requested_side_count,
        "reachable_side_quests": [list(cell) for cell in env.side_quests_reached],
        "reachable_side_count": env.side_count,
        "main_reachable": env.main_reached,
        "solved": env.is_solved(),
        "edits": env.edits_made,
        "heightmap": env.heightmap.astype(int).tolist(),
        "original_heightmap": record["heightmap"].astype(int).tolist(),
    }
    with open(os.path.join(folder, f"{record['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
