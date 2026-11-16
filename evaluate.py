import json
import os
import sys

import gymnasium as gym
import torch

from heightmap_env import HeightmapEnv, load_maps, split_maps
from ppo_train import Agent, format_eval, play_map_greedily, score_eval

'''
Plays every test map with a trained agent and writes the edited maps to maps_edited/.
Usage: python evaluate.py runs/<run_name>/agent.pt [maps folder] [seed]
The seed must be the one used in training, so the held-out maps are the same ones.
'''
def main():
    if len(sys.argv) < 2:
        print("usage: python evaluate.py runs/<run_name>/agent.pt [maps folder] [seed]")
        return
    checkpoint_path = sys.argv[1]
    maps_folder = sys.argv[2] if len(sys.argv) > 2 else "maps"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps_edited")
    os.makedirs(output_folder, exist_ok=True)
    for name in os.listdir(output_folder):
        if name.endswith(".json") and name[:-5].isdigit():
            os.remove(os.path.join(output_folder, name))

    _, test_records = split_maps(load_maps(maps_folder), seed, eval_fraction=0.2)
    print("test maps:", ", ".join(str(record["id"]) for record in test_records))
    env = HeightmapEnv(test_records, fixed_order=True)

    # Agent only reads the observation and action sizes from this
    shape_source = gym.vector.SyncVectorEnv([lambda: HeightmapEnv(test_records)])
    agent = Agent(shape_source)
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.eval()

    results = []
    print(f"{'map':>4} {'start':>5} {'end':>4} {'want':>4} {'main':>5} {'solved':>6} {'edits':>5}")
    for record in test_records:
        result = play_map_greedily(agent, env, torch.device("cpu"))
        results.append(result)
        print(
            f"{record['id']:>4} {env.start_side_count:>5} {result['side_reached']:>4} {result['side_target']:>4} "
            f"{str(result['main_reachable']):>5} {str(result['terminated']):>6} {result['length']:>5}"
        )
        save_edited_map(output_folder, record, env)

    print()
    print(format_eval(results, score_eval(results)))
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
