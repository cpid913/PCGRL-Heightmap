import argparse
import json
import os
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from heightmap import generate_heightmap
from path_finding import floodfill
from quest_placement import place_quests

MAP_SIZE = 16
HOW_MANY = 100
MAX_TRIES = 20000

'''
Folder next to this file. Created if it is missing.
'''
def maps_dir():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps_new")
    os.makedirs(folder, exist_ok=True)
    return folder

'''
One try. requested_reachable is how many side quests we hoped the player could reach.
It is drawn after the quests are placed so it never asks for more side quests than exist.
'''
def roll_map(seed):
    rng = np.random.default_rng(seed)
    heightmap = generate_heightmap(MAP_SIZE, rng)

    # place_quests prints, and most of these maps are thrown away
    with redirect_stdout(StringIO()):
        spawn, main_quest, side_quests = place_quests(heightmap, rng)
    wanted = int(rng.integers(0, len(side_quests) + 1))

    walked = floodfill(heightmap, spawn)
    reached = [cell for cell in side_quests if cell in walked]
    main_reachable = main_quest in walked
    return heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable

'''
Quest fields first. The heightmap is the long part so it goes last.
'''
def save_map(folder, number, seed, heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable):
    record = {
        "id": number,
        "seed": int(seed),
        "spawn": list(spawn),
        "main_quest": list(main_quest),
        "side_quests": [list(cell) for cell in side_quests],
        "requested_reachable": wanted,
        "reachable_side_quests": [list(cell) for cell in reached],
        "reachable_side_count": len(reached),
        "main_reachable": main_reachable,
        "heightmap": heightmap.astype(int).tolist(),
    }
    path = os.path.join(folder, f"{number}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path

'''
Numbered map files already in the folder: (highest id, highest seed used), or (0, -1) if there are none.
'''
def existing_maps(folder):
    last_id, last_seed = 0, -1
    for name in os.listdir(folder):
        if name.endswith(".json") and name[:-5].isdigit():
            last_id = max(last_id, int(name[:-5]))
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                last_seed = max(last_seed, int(json.load(f)["seed"]))
    return last_id, last_seed

'''
Keep rolling until how_many new maps have an unreachable main quest.
New maps are numbered after first_id, so with an existing batch of 1..100 they become 101.json, 102.json, ...
'''
def collect(folder, how_many, start_seed=0, first_id=0):
    kept = 0
    seed = start_seed
    tries = 0

    while kept < how_many and tries < MAX_TRIES:
        heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable = roll_map(seed)
        tries += 1

        if not main_reachable:
            kept += 1
            save_map(folder, first_id + kept, seed, heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable)
            print(
                f"kept {kept}/{how_many}  id {first_id + kept}  seed {seed}  "
                f"requested {wanted}  reached {len(reached)}  "
                f"sides placed {len(side_quests)}"
            )
        elif tries % 200 == 0:
            print(f"tried {tries}, kept {kept}")

        seed += 1

    return kept, tries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=HOW_MANY, help="how many new maps to add")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the existing numbered maps first and start again from 1.json / seed 0")
    args = parser.parse_args()

    folder = maps_dir()

    if args.fresh:
        for name in os.listdir(folder):
            if name.endswith(".json") and name[:-5].isdigit():
                os.remove(os.path.join(folder, name))

    # Append: continue the numbering and start after the highest seed already used, so no map repeats
    last_id, last_seed = existing_maps(folder)
    kept, tries = collect(folder, args.count, start_seed=last_seed + 1, first_id=last_id)
    print(f"Wrote {kept} maps ({last_id + 1}..{last_id + kept}) into {folder} after {tries} tries")
    if kept < args.count:
        print(f"Stopped early: only {kept} maps matched in {MAX_TRIES} tries")
