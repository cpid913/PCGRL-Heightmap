import gymnasium as gym
import numpy as np
from path_finding import floodfill, floodfill_stop
from main import show_heightmap
'''
Environment class for pqn (written based on a template code generated from Claude)
Used Claude for debugging
'''
class PQNEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maps, max_steps=500):
        super().__init__()
        self.maps = maps
        self.observation_space = gym.spaces.Box(low=0, high=8, shape=(6,16,16), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(512)
        self.max_steps = max_steps
        self.t = 0

    '''
    Observation includes: current heightmap, location of the spawn, location of the main quest, location of the side quests, target number of reached side quests and exploration area
    '''
    def _obs(self):
        return np.array([self.current_heightmap, self.mask_spawn, self.mask_main_quest, self.mask_side_quests, self.demand_map, self.exploration_area], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # options={"map_idx": i} picks a specific map (used for evaluation); otherwise pick at random
        if options is not None and "map_idx" in options:
            idx = options["map_idx"]
        else:
            idx = self.np_random.integers(len(self.maps))
        self.current_map = self.maps[idx].copy()
        # Store initial map for fianl comparison purpose
        # Copy the heightmap so step() edits don't overwrite the original loaded map
        self.current_heightmap = self.current_map["heightmap"].copy()
        self.initial_heightmap = self.current_heightmap.copy()
        self.spawn = self.current_map["spawn"]
        self.main_quest = self.current_map["main_quest"]
        self.side_quests = self.current_map["side_quests"]

        mask_spawn = np.zeros_like(self.current_heightmap, dtype=np.float32)
        mask_spawn[self.current_map["spawn"][0], self.current_map["spawn"][1]] = 1
        self.mask_spawn = mask_spawn

        mask_main_quest = np.zeros_like(self.current_heightmap, dtype=np.float32)
        mask_main_quest[self.current_map["main_quest"][0], self.current_map["main_quest"][1]] = 1
        self.mask_main_quest = mask_main_quest

        mask_side_quests = np.zeros_like(self.current_heightmap, dtype=np.float32)
        if len(self.current_map["side_quests"]) > 0:
            xs, ys = zip(*self.current_map["side_quests"])
            mask_side_quests[list(xs), list(ys)] = 1
        self.mask_side_quests = mask_side_quests

        self.target_side_quests = self.current_map["demand"]
        self.t = 0

        visited = floodfill_stop(self.current_heightmap, self.spawn, self.main_quest)
        exploration_area = np.zeros_like(self.current_heightmap, dtype=np.float32)
        xs, ys = zip(*visited)
        exploration_area[list(xs), list(ys)] = 1
        self.exploration_area = exploration_area
        self.initial_visited = visited
        reached_sides = [cell for cell in self.side_quests if cell in visited]
        self.current_diff_side_quests = abs(self.target_side_quests - len(reached_sides))
        self.current_main_reachable = self.current_map["main_reached"]
        self.demand_map = np.full((16, 16), self.target_side_quests)
        return self._obs(), {"map_idx": int(idx)}

    def step(self, action):
        # Each action in [0, 255] will add one height to the corresponding location, and each action in [256, 512] will lower one height to the corresponding location
        # The height must be within [1, 8]. An invalid action will not change the map.
        if action <= 255:
            if self.current_heightmap[np.floor(action / 16).astype(int)][action % 16] < 8:
                self.current_heightmap[np.floor(action / 16).astype(int)][action % 16] += 1
        else:
            if self.current_heightmap[np.floor((action - 256) / 16).astype(int)][(action - 256) % 16] > 1:
                self.current_heightmap[np.floor((action - 256) / 16).astype(int)][(action - 256) % 16] -= 1

        self.t += 1

        visited = floodfill_stop(self.current_heightmap, self.spawn, self.main_quest)
        # Update exploration area
        exploration_area = np.zeros_like(self.current_heightmap, dtype=np.float32)
        xs, ys = zip(*visited)
        exploration_area[list(xs), list(ys)] = 1
        self.exploration_area = exploration_area

        reached_sides = [cell for cell in self.side_quests if cell in visited]
        main_reachable = float(self.main_quest in visited)
        diff_side = abs(self.target_side_quests - len(reached_sides))
        # Start calculating reward
        # Small penalty at each step to encourage fewer steps
        reward = -0.1

        # Reward if the number of side quests is closer to the target, penalize if the number of side quests is farther than the target
        reward += (self.current_diff_side_quests - diff_side)
        self.current_diff_side_quests = diff_side

        # Reward if the current action made main quest reachable, penalize if the current action made main quest unreachable
        if main_reachable != self.current_main_reachable:
            reward += 10 if main_reachable else -10
            self.current_main_reachable = main_reachable
        terminated = (abs(self.target_side_quests - len(reached_sides)) == 0) and (self.main_quest in visited)
        if terminated:
            reward += 10
        truncated = self.t >= self.max_steps

        info = {
            "main_reachable": bool(main_reachable),
            "side_reached": len(reached_sides),
            "side_target": self.target_side_quests,
        }
        return self._obs(), reward, terminated, truncated, info