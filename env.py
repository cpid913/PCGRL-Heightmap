import gymnasium as gym
import numpy as np
from path_finding import floodfill, floodfill_stop
class PQNEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maps, max_steps=200):
        super().__init__()
        self.maps = maps
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,16,16), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(512)
        self.max_steps = max_steps
        self.t = 0

    '''
    Observation includes: current heightmap, location of the spawn, location of the main quest, location of the side quests
    '''
    def _obs(self):
        return np.array([self.current_heightmap, self.mask_spawn, self.mask_main_quest, self.mask_side_quests], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.maps))
        self.current_map = self.maps[idx].copy()
        self.current_heightmap = self.current_map["heightmap"]
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
        reached_sides = [cell for cell in self.side_quests if cell in visited]
        # 1 if the main quest is not reachable, 0 if reachable
        main_reachable = float(not (self.main_quest in visited))
        reward = - abs(self.target_side_quests - len(reached_sides)) - 10 * main_reachable
        terminated = (abs(self.target_side_quests - len(reached_sides)) == 0) and (self.main_quest in visited)
        if terminated:
            reward += 10
        truncated = self.t >= self.max_steps   # time limit, not a real terminal

        return self._obs(), reward, terminated, truncated, {}