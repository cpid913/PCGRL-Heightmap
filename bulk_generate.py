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
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    os.makedirs(folder, exist_ok=True)
    return folder

'''
One try. Side quests and the requested reachable count are each random 0 to 5.
requested_reachable is how many side quests we hoped the player could reach.
'''
def roll_map(seed):
    rng = np.random.default_rng(seed)
    heightmap = generate_heightmap(MAP_SIZE, rng)
    wanted = int(rng.integers(0, 6))

    # place_quests prints, and most of these maps are thrown away
    with redirect_stdout(StringIO()):
        spawn, main_quest, side_quests = place_quests(heightmap, rng)

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
Keep rolling until HOW_MANY maps have an unreachable main quest.
Each kept map is maps/1.json, maps/2.json, and so on.
'''
def collect(folder, start_seed=0):
    kept = 0
    seed = start_seed
    tries = 0

    while kept < HOW_MANY and tries < MAX_TRIES:
        heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable = roll_map(seed)
        tries += 1

        if not main_reachable:
            kept += 1
            save_map(folder, kept, seed, heightmap, spawn, main_quest, side_quests, wanted, reached, main_reachable)
            print(
                f"kept {kept}/{HOW_MANY}  seed {seed}  "
                f"requested {wanted}  reached {len(reached)}  "
                f"sides placed {len(side_quests)}"
            )
        elif tries % 200 == 0:
            print(f"tried {tries}, kept {kept}")

        seed += 1

    return kept, tries

if __name__ == "__main__":
    folder = maps_dir()

    # Start the numbering at 1 again. Old numbered files are the previous batch.
    for name in os.listdir(folder):
        if name.endswith(".json") and name[:-5].isdigit():
            os.remove(os.path.join(folder, name))

    kept, tries = collect(folder)
    print(f"Wrote {kept} maps into {folder} after {tries} tries")
    if kept < HOW_MANY:
        print(f"Stopped early: only {kept} maps matched in {MAX_TRIES} tries")
