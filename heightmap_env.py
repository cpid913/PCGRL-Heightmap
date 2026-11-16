import json
import os

import gymnasium as gym
import numpy as np

from heightmap import MAX_HEIGHT
from path_finding import floodfill_stop

MAP_SIZE = 16
MAX_SIDE_QUESTS = 5
MAX_EDITS = 500
STEP_COST = 0.02
SUCCESS_BONUS = 2.0  # paid once on the edit that solves the map, so finishing beats fiddling
LAST_TRAIN_ID = 80  # maps 1..80 train, 81..100 test

LOWER = 0
RAISE = 1


'''
Reads maps/<id>.json and keeps the train ids or the test ids.
Maps asking for more side quests than were placed can never be solved, so they are skipped.
'''
def load_split(folder, split):
    records = []
    skipped = 0
    for name in os.listdir(folder):
        if not (name.endswith(".json") and name[:-5].isdigit()):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            raw = json.load(f)

        in_train = raw["id"] <= LAST_TRAIN_ID
        if in_train != (split == "train"):
            continue
        if raw["requested_reachable"] > len(raw["side_quests"]):
            skipped += 1
            continue

        records.append({
            "id": raw["id"],
            "seed": raw["seed"],
            "heightmap": np.array(raw["heightmap"], dtype=np.int32),
            "spawn": tuple(raw["spawn"]),
            "main_quest": tuple(raw["main_quest"]),
            "side_quests": [tuple(cell) for cell in raw["side_quests"]],
            "requested_side_count": raw["requested_reachable"],
        })
    records.sort(key=lambda record: record["id"])
    if skipped:
        print(f"{split}: skipped {skipped} maps that ask for more side quests than were placed")
    return records


class HeightmapEnv(gym.Env):
    '''
    The agent raises or lowers one column per step.
    Action a: direction = a // 256 (0 lower, 1 raise), cell = a % 256, row, col = divmod(cell, 16).
    Done when the main quest is reachable and exactly the requested number of side quests are.
    '''

    def __init__(self, records, fixed_order=False):
        self.records = records
        # Evaluation walks the maps in order instead of picking them at random
        self.fixed_order = fixed_order
        self.next_record_index = 0

        self.action_space = gym.spaces.Discrete(2 * MAP_SIZE * MAP_SIZE)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (7, MAP_SIZE, MAP_SIZE), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_order:
            self.record = self.records[self.next_record_index % len(self.records)]
            self.next_record_index += 1
        else:
            self.record = self.records[self.np_random.integers(len(self.records))]

        self.heightmap = self.record["heightmap"].copy()
        self.spawn = self.record["spawn"]
        self.main_quest = self.record["main_quest"]
        self.side_quests = self.record["side_quests"]
        self.requested_side_count = self.record["requested_side_count"]
        self.edits_made = 0

        self.update_reachability()
        self.start_side_count = self.side_count
        return self.observation(), self.info()

    def step(self, action):
        direction, cell = divmod(int(action), MAP_SIZE * MAP_SIZE)
        row, col = divmod(cell, MAP_SIZE)
        change = 1 if direction == RAISE else -1
        self.heightmap[row, col] = np.clip(self.heightmap[row, col] + change, 1, MAX_HEIGHT)
        self.edits_made += 1

        side_miss_before = abs(self.side_count - self.requested_side_count)
        main_before = self.main_reached
        self.update_reachability()
        side_miss_after = abs(self.side_count - self.requested_side_count)

        # Reaching too many side quests is punished the same as reaching too few
        reward = (side_miss_before - side_miss_after) + (int(self.main_reached) - int(main_before)) - STEP_COST

        solved = self.is_solved()
        if solved:
            reward += SUCCESS_BONUS
        out_of_edits = self.edits_made >= MAX_EDITS and not solved
        return self.observation(), float(reward), solved, out_of_edits, self.info()

    def update_reachability(self):
        # Spreads out from spawn and stops at the step it reaches the main quest,
        # so a side quest counts only if the player gets there no later than the main quest
        self.reachable_cells = floodfill_stop(self.heightmap, self.spawn, self.main_quest)
        self.side_quests_reached = [cell for cell in self.side_quests if cell in self.reachable_cells]
        self.side_count = len(self.side_quests_reached)
        self.main_reached = self.main_quest in self.reachable_cells

    def is_solved(self):
        return self.main_reached and self.side_count == self.requested_side_count

    def observation(self):
        planes = np.zeros((7, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        planes[0] = self.heightmap / MAX_HEIGHT
        planes[1][self.spawn] = 1.0
        planes[2][self.main_quest] = 1.0
        for cell in self.side_quests:
            planes[3][cell] = 1.0
        for cell in self.reachable_cells:
            planes[4][cell] = 1.0
        for cell in self.side_quests_reached:
            planes[5][cell] = 1.0
        planes[6][:] = self.requested_side_count / MAX_SIDE_QUESTS
        return planes

    def action_mask(self):
        # Heights stay inside 1..MAX_HEIGHT, so a column at the floor cannot go lower and one at the top cannot go higher
        can_lower = (self.heightmap > 1).ravel()
        can_raise = (self.heightmap < MAX_HEIGHT).ravel()
        return np.concatenate([can_lower, can_raise])

    def info(self):
        return {
            "action_mask": self.action_mask(),
            "success": self.is_solved(),
            "side_count": self.side_count,
            "requested_side_count": self.requested_side_count,
            "main_reached": self.main_reached,
            "edits_made": self.edits_made,
            "map_id": self.record["id"],
        }
