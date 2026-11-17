"""
Written by Claude Code:
Greedy evaluation for the PQN heightmap agent.

Used two ways:
  1. Imported by pqn.py for the final evaluation after training.
  2. Run on its own, while training is still going, on the checkpoints pqn.py saves:
       python evaluate_pqn.py runs/<run_name>             # evaluate all checkpoints so far
       python evaluate_pqn.py runs/<run_name> --watch     # keep evaluating new checkpoints until training ends
       python evaluate_pqn.py runs/<run_name> --model best_model.pt   # evaluate one saved model (per-map results)
     Results are printed and logged to TensorBoard under runs/<run_name>/eval_offline.
"""
import glob
import json
import os
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import tyro

from env import PQNEnv
from path_finding import floodfill_stop


def valid_action_mask(obs):
    """obs: (B, C, 16, 16) tensor, channel 0 = raw heightmap (1..8).
    Returns a (B, 512) bool mask: actions 0-255 raise a cell (valid if height < 8),
    256-511 lower a cell (valid if height > 1). Row-major, so action a -> cell (a // 16, a % 16),
    matching PQNEnv.step. If the heightmap channel is ever rescaled, update these thresholds."""
    h = obs[:, 0].reshape(obs.shape[0], -1)
    return torch.cat([h < 8, h > 1], dim=1)


def masked_q(q_values, mask):
    """Invalid actions get -inf so max/argmax only ever consider valid actions."""
    return q_values.masked_fill(~mask, float("-inf"))


def load_pqn_maps():
    """The maps in the format PQNEnv expects. Reuses map_browser.load_maps and adds the two keys
    the env needs: 'heightmap' (the map's terrain) and 'main_reached' (main quest reachable at the start)."""
    from map_browser import load_maps
    maps = []
    for item in load_maps():
        m = dict(item)
        m["heightmap"] = item["edited_heightmap"]  # for maps/ this is just the map's heightmap
        m["main_reached"] = m["main_quest"] in floodfill_stop(m["heightmap"], m["spawn"], m["main_quest"])
        maps.append(m)
    return maps


def split_maps(maps, seed, eval_fraction):
    """Train/eval split: shuffle once with the run seed, hold out eval_fraction of the maps."""
    perm = np.random.default_rng(seed).permutation(len(maps))
    num_eval = max(1, int(round(len(maps) * eval_fraction)))
    eval_maps = [maps[i] for i in perm[:num_eval]]
    train_maps = [maps[i] for i in perm[num_eval:]]
    return train_maps, eval_maps


def evaluate(q_network, eval_maps, device):
    """Run one greedy (epsilon = 0) episode on every eval map. Returns a list of per-map results.
    All maps are played at the same time (one env per map), so each step is a single batched
    forward pass instead of one tiny GPU call per map per step."""
    envs = [PQNEnv(eval_maps) for _ in eval_maps]
    obs = np.stack([env.reset(options={"map_idx": i})[0] for i, env in enumerate(envs)])
    returns = [0.0] * len(envs)
    lengths = [0] * len(envs)
    results = [None] * len(envs)
    active = list(range(len(envs)))  # maps whose episode is still running
    was_training = q_network.training
    q_network.eval()
    while active:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs[active]).to(device)
            q_values = masked_q(q_network(obs_t), valid_action_mask(obs_t))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        still_active = []
        for i, action in zip(active, actions):
            env = envs[i]
            obs[i], reward, terminated, truncated, info = env.step(int(action))
            returns[i] += reward
            lengths[i] += 1
            if not (terminated or truncated):
                still_active.append(i)
                continue
            # Final state of the episode
            results[i] = {
                "map_id": eval_maps[i]["id"],
                "return": returns[i],
                "length": lengths[i],
                "terminated": bool(terminated),
                "main_reachable": info["main_reachable"],
                "side_reached": info["side_reached"],
                "side_target": info["side_target"],
                # signed: > 0 means too many side quests reachable, < 0 means too few
                "side_error": info["side_reached"] - info["side_target"],
                # How much the map was edited: initial vs final heightmap
                "cells_changed": int((env.current_heightmap != env.initial_heightmap).sum()),
                "height_change": int(np.abs(env.current_heightmap.astype(int) - env.initial_heightmap.astype(int)).sum()),
                # Split of height_change: blocks added (cells that ended higher) and removed (cells that ended lower)
                "height_added": int(np.clip(env.current_heightmap.astype(int) - env.initial_heightmap.astype(int), 0, None).sum()),
                "height_removed": int(np.clip(env.initial_heightmap.astype(int) - env.current_heightmap.astype(int), 0, None).sum()),
                # The edited map itself, for view_policy.py
                "final_heightmap": env.current_heightmap.copy(),
            }
        active = still_active
    q_network.train(was_training)
    return results


def log_eval(writer, results, global_step, per_map=False):
    """Print the summary and write eval/* scalars. per_map also logs one point per map (x = map index)."""
    if per_map:
        for i, r in enumerate(results):
            writer.add_scalar("eval/episodic_return", r["return"], i)
            writer.add_scalar("eval/episodic_length", r["length"], i)
            writer.add_scalar("eval/terminated", int(r["terminated"]), i)
            writer.add_scalar("eval/main_reachable", int(r["main_reachable"]), i)
            writer.add_scalar("eval/side_error", r["side_error"], i)
    success_rate = np.mean([r["terminated"] for r in results])
    main_success_rate = np.mean([r["main_reachable"] for r in results])
    side_success_rate = np.mean([r["side_error"] == 0 for r in results])
    # MAE over all episodes: mean |reached - requested| side quests
    side_mae = np.mean([abs(r["side_error"]) for r in results])
    mean_return = np.mean([r["return"] for r in results])
    mean_length = np.mean([r["length"] for r in results])
    writer.add_scalar("eval/success_rate", success_rate, global_step)
    writer.add_scalar("eval/main_success_rate", main_success_rate, global_step)
    writer.add_scalar("eval/side_mae", side_mae, global_step)
    mean_cells_changed = np.mean([r["cells_changed"] for r in results])
    mean_height_change = np.mean([r["height_change"] for r in results])
    # Totals over all eval maps (added + removed = total height change)
    total_height_added = sum(r["height_added"] for r in results)
    total_height_removed = sum(r["height_removed"] for r in results)
    print(f"EVAL step {global_step} on {len(results)} held-out maps: success_rate={success_rate:.2%}, "
          f"main_success_rate={main_success_rate:.2%}, side_success_rate={side_success_rate:.2%}, side_mae={side_mae:.2f}, "
          f"mean_return={mean_return:.2f}, mean_length={mean_length:.1f}, "
          f"mean_cells_changed={mean_cells_changed:.1f}, mean_height_change={mean_height_change:.1f}, "
          f"total_height_added={total_height_added}, total_height_removed={total_height_removed}")


# ---------------------------------------------------------------- standalone use

@dataclass
class EvalArgs:
    run_dir: tyro.conf.Positional[str]
    """the run folder, e.g. runs/Heightmap__pqn__1__1794784219"""
    watch: bool = False
    """keep polling for new checkpoints until the final model.pt appears"""
    poll_seconds: int = 30
    """how often to look for new checkpoints when --watch is on"""
    cuda: bool = False
    """evaluate on GPU (off by default so it doesn't compete with training)"""
    seed: int = 1
    """only used if the run has no eval_maps.json: must match the training --seed"""
    eval_fraction: float = 0.2
    """only used if the run has no eval_maps.json: must match the training --eval-fraction"""
    model: str = ""
    """evaluate just this one file in run_dir instead of the checkpoints, e.g. --model best_model.pt"""


def checkpoint_step(path):
    match = re.search(r"step_(\d+)\.pt$", path)
    return int(match.group(1)) if match else None


def load_eval_maps(args):
    maps = load_pqn_maps()
    ids_path = os.path.join(args.run_dir, "eval_maps.json")
    if os.path.exists(ids_path):
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)["eval_map_ids"]
        by_id = {m["id"]: m for m in maps}
        return [by_id[i] for i in ids]
    print(f"no eval_maps.json in {args.run_dir}, recomputing the split from seed={args.seed}")
    return split_maps(maps, args.seed, args.eval_fraction)[1]


def evaluate_single(args, q_network, eval_maps, device, SummaryWriter):
    """--model: evaluate one saved file (e.g. best_model.pt) and print per-map results plus the summary.
    TensorBoard output goes to runs/<run>/eval_offline/<file name>, so it doesn't mix with checkpoint evals."""
    path = os.path.join(args.run_dir, args.model)
    q_network.load_state_dict(torch.load(path, map_location=device))
    # best_model.json says which training step best_model.pt came from
    step = 0
    info_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            step = json.load(f).get("step") or 0
        print(f"{args.model} is from training step {step}")
    results = evaluate(q_network, eval_maps, device)
    for r in results:
        print(f"  map {r['map_id']}: terminated={r['terminated']}, main_reachable={r['main_reachable']}, "
              f"side {r['side_reached']}/{r['side_target']}, length={r['length']}, return={r['return']:.2f}, "
              f"added={r['height_added']}, removed={r['height_removed']}")
    writer = SummaryWriter(os.path.join(args.run_dir, "eval_offline", os.path.splitext(os.path.basename(args.model))[0]))
    log_eval(writer, results, step, per_map=True)
    writer.close()


def main(args):
    # Imported here (not at the top) because pqn.py imports this module
    from torch.utils.tensorboard import SummaryWriter
    from pqn import QNetwork

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    eval_maps = load_eval_maps(args)
    probe = PQNEnv(eval_maps)
    spaces = SimpleNamespace(single_observation_space=probe.observation_space,
                             single_action_space=probe.action_space)
    q_network = QNetwork(spaces).to(device)

    if args.model:
        evaluate_single(args, q_network, eval_maps, device, SummaryWriter)
        return

    writer = SummaryWriter(os.path.join(args.run_dir, "eval_offline"))
    final_path = os.path.join(args.run_dir, "model.pt")

    done_steps = set()
    while True:
        training_finished = os.path.exists(final_path)
        paths = glob.glob(os.path.join(args.run_dir, "checkpoints", "step_*.pt"))
        pending = sorted((checkpoint_step(p), p) for p in paths if checkpoint_step(p) not in done_steps)
        for step, path in pending:
            try:
                q_network.load_state_dict(torch.load(path, map_location=device))
            except Exception as e:  # e.g. the file is still being written; retry next poll
                print(f"could not load {path} yet ({e})")
                continue
            log_eval(writer, evaluate(q_network, eval_maps, device), step)
            writer.flush()
            done_steps.add(step)

        if not args.watch or training_finished:
            break
        time.sleep(args.poll_seconds)

    if not done_steps and os.path.exists(final_path):
        # Older runs saved only the final model
        q_network.load_state_dict(torch.load(final_path, map_location=device))
        log_eval(writer, evaluate(q_network, eval_maps, device), 0, per_map=True)
    elif not done_steps:
        print(f"no checkpoints or model.pt found in {args.run_dir}")
    writer.close()


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
