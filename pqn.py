"""
Code from https://github.com/vwxyzjn/cleanrl with modification to fit this project
Used Claude for debugging
"""
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqnpy
import json
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from env import PQNEnv
from evaluate_pqn import evaluate, load_pqn_maps, log_eval, masked_q, split_maps, valid_action_mask

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    split_seed: int = 67
    """seed for train/valid split"""
    seed: int = 1
    """seed of the training experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Heightmap"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run for each environment per update"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    anneal_lr: bool = True
    """Toggle learning rate annealing"""
    gamma: float = 0.99
    """the discount factor gamma"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    q_lambda: float = 0.65
    """the lambda for Q(lambda)"""
    eval_fraction: float = 0.2
    """fraction of maps held out for evaluation after training"""
    checkpoint_every: int = 50000
    """save a checkpoint to runs/<run>/checkpoints every this many timesteps (0 = only at the end)"""
    eval_every: int = 10000
    """run a greedy evaluation on the held-out maps every this many timesteps and log eval/* (0 = only at the end)"""


def make_env(env_id, maps, seed, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.action_space.seed(seed)
        env = PQNEnv(maps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

'''
CNN built with Claude
'''
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        C, H, W = env.single_observation_space.shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * H * W, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.single_action_space.n)),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def crossed(global_step, every, batch_size):
    """True once each time global_step passes a multiple of `every` (steps advance by batch_size per iteration)."""
    return every > 0 and global_step // every > (global_step - batch_size) // every


def save_atomic(q_network, path):
    """Write to a temp file first so evaluate.py --watch never reads a half-written checkpoint."""
    torch.save(q_network.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def save_checkpoint(q_network, run_name, global_step):
    folder = f"runs/{run_name}/checkpoints"
    os.makedirs(folder, exist_ok=True)
    save_atomic(q_network, os.path.join(folder, f"step_{global_step}.pt"))


def save_if_best(q_network, run_name, results, global_step, best):
    """Save runs/<run>/best_model.pt when this evaluation beats the best so far, ranked by:
    1. higher success_rate, 2. higher main_success_rate, 3. lower side_mae; if all three tie, the earlier model is kept.
    `best` is a dict carried across evaluations; best_model.json records which step it came from."""
    success_rate = float(np.mean([r["terminated"] for r in results]))
    main_success_rate = float(np.mean([r["main_reachable"] for r in results]))
    side_mae = float(np.mean([abs(r["side_error"]) for r in results]))
    # Tuples compare field by field; side_mae is negated so that "bigger is better" holds for every field
    if (success_rate, main_success_rate, -side_mae) <= (best["success_rate"], best["main_success_rate"], -best["side_mae"]):
        return
    best.update(success_rate=success_rate, main_success_rate=main_success_rate, side_mae=side_mae, step=global_step)
    save_atomic(q_network, f"runs/{run_name}/best_model.pt")
    with open(f"runs/{run_name}/best_model.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"new best model at step {global_step}: success_rate={success_rate:.2%}, "
          f"main_success_rate={main_success_rate:.2%}, side_mae={side_mae:.2f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Train/eval split: shuffle once with the run seed, hold out eval_fraction of the maps
    train_maps, eval_maps = split_maps(load_pqn_maps(), args.split_seed, args.eval_fraction)
    print(f"maps: {len(train_maps)} train / {len(eval_maps)} eval")
    writer.add_text("eval_map_ids", ", ".join(str(m["id"]) for m in eval_maps))
    # So evaluate.py can use exactly the same held-out maps
    with open(f"runs/{run_name}/eval_maps.json", "w", encoding="utf-8") as f:
        json.dump({"eval_map_ids": [int(m["id"]) for m in eval_maps]}, f)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, train_maps, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent setup
    q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    eval_seconds = 0.0
    # Best evaluation so far; worst-possible start values so the first evaluation always saves a best_model.pt
    best = {"success_rate": -1.0, "main_success_rate": -1.0, "side_mae": float("inf"), "step": None}
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            # Mask is recomputed every step from the current heightmap in the observation
            mask = valid_action_mask(next_obs)
            # Explore uniformly over valid actions only
            random_actions = torch.multinomial(mask.float(), 1).squeeze(1)
            with torch.no_grad():
                q_values = masked_q(q_network(next_obs), mask)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, "
                              f"terminated={terminations[env_idx]}, truncated={truncations[env_idx]}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/terminated", int(terminations[env_idx]), global_step)

                        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # Bootstrap with the best *valid* action; other steps reuse values[t + 1], already masked
                    next_value, _ = torch.max(masked_q(q_network(next_obs), valid_action_mask(next_obs)), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        # SPS excludes time spent in periodic evaluation, so it still measures training speed
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time - eval_seconds)), global_step)

        # Periodic checkpoint for evaluate.py (fires once each time global_step crosses a multiple)
        if crossed(global_step, args.checkpoint_every, args.batch_size):
            save_checkpoint(q_network, run_name, global_step)

        # Periodic greedy evaluation on the held-out maps -> eval/main_success_rate, eval/side_mae, ... vs global_step
        # (skipped on the last iteration: the final evaluation below logs that step)
        if crossed(global_step, args.eval_every, args.batch_size) and iteration < args.num_iterations:
            eval_start = time.time()
            results = evaluate(q_network, eval_maps, device)
            log_eval(writer, results, global_step)
            save_if_best(q_network, run_name, results, global_step, best)
            eval_seconds += time.time() - eval_start

    envs.close()

    # Final checkpoint first, then model.pt (evaluate.py --watch stops once model.pt exists)
    save_checkpoint(q_network, run_name, global_step)
    model_path = f"runs/{run_name}/model.pt"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # Evaluate on held-out maps
    results = evaluate(q_network, eval_maps, device)
    log_eval(writer, results, global_step, per_map=True)
    save_if_best(q_network, run_name, results, global_step, best)
    print(f"best model: step {best['step']}, success_rate={best['success_rate']:.2%}, "
          f"main_success_rate={best['main_success_rate']:.2%}, side_mae={best['side_mae']:.2f} -> runs/{run_name}/best_model.pt")

    writer.close()
