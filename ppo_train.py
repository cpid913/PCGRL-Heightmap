# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# Copied from CleanRL's ppo.py (untouched copy in cleanrl_ppo_original.py). Changes for the heightmap are marked "heightmap:".
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# heightmap: our environment instead of a registered gym one
from heightmap_env import HeightmapEnv, load_maps, split_maps


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False  # heightmap: training on CPU
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
    env_id: str = "Heightmap-v0"  # heightmap: only used in the run name
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # heightmap: where the maps live and how often to try the test maps
    maps_folder: str = "maps"
    """folder with the numbered map json files"""
    eval_fraction: float = 0.2
    """fraction of maps held out for testing, shuffled with the run seed"""
    eval_every: int = 50_000
    """run the test maps every this many steps and keep the best model"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# heightmap: build our env from the training maps; there is nothing to record on video
def make_env(train_records):
    def thunk():
        env = HeightmapEnv(train_records)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # heightmap: the map is a 16x16 grid, so a small CNN replaces the MLPs. Padding keeps every layer 16x16.
        # Kept at 32 channels so training runs at a usable speed on CPU
        channels, height, width = envs.single_observation_space.shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding=1)),
            nn.ReLU(),
        )
        # heightmap: squeeze to 8 channels first, otherwise the linear layer is huge and slow
        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(32, 8, 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(8 * height * width, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        # heightmap: one "lower" and one "raise" score per cell; flattening gives the env's action numbering
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(32, 2, 1), std=0.01),
            nn.Flatten(),
        )

    def get_value(self, x):
        return self.critic(self.network(x))

    # heightmap: edits that would push a column outside 1..8 get a huge negative score so they are never picked
    def masked_logits(self, x, action_mask):
        logits = self.actor(self.network(x))
        return torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))

    def get_action_and_value(self, x, action=None, action_mask=None):
        logits = self.masked_logits(x, action_mask)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


# heightmap: play one map with the most likely legal edit each step, no sampling
def play_map_greedily(agent, env, device):
    observation, info = env.reset()
    episodic_return = 0.0
    finished = False
    terminated = False
    while not finished:
        with torch.no_grad():
            observation_tensor = torch.Tensor(observation).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(info["action_mask"]).unsqueeze(0).to(device)
            action = agent.masked_logits(observation_tensor, mask_tensor).argmax(dim=1).item()
        observation, reward, terminated, truncated, info = env.step(action)
        episodic_return += reward
        finished = terminated or truncated
    # Same fields pqn.evaluate records for one map
    return {
        "map_id": info["map_id"],
        "return": episodic_return,
        "length": info["edits_made"],
        "terminated": bool(terminated),
        "main_reachable": bool(info["main_reached"]),
        "side_reached": info["side_count"],
        "side_target": info["requested_side_count"],
        "side_error": info["side_count"] - info["requested_side_count"],
    }


# heightmap: one greedy episode on every test map
def evaluate_on_test_maps(agent, test_records, device):
    was_training = agent.training
    agent.eval()
    env = HeightmapEnv(test_records, fixed_order=True)
    results = [play_map_greedily(agent, env, device) for _ in test_records]
    agent.train(was_training)
    return results


# heightmap: same summary pqn.py prints after its held-out eval
def score_eval(results):
    scores = {
        "success_rate": float(np.mean([r["terminated"] for r in results])),
        "main_success_rate": float(np.mean([r["main_reachable"] for r in results])),
        "side_success_rate": float(np.mean([r["side_error"] == 0 for r in results])),
        "side_mae": float(np.mean([abs(r["side_error"]) for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
    }
    failed = [r for r in results if not r["terminated"]]
    if failed:
        errors = np.array([r["side_error"] for r in failed])
        scores["failed_side_abs_error"] = float(np.abs(errors).mean())
        scores["failed_side_signed_error"] = float(errors.mean())
    return scores


def format_eval(results, scores):
    return (
        f"EVAL on {len(results)} held-out maps: success_rate={scores['success_rate']:.2%}, "
        f"main_success_rate={scores['main_success_rate']:.2%}, side_success_rate={scores['side_success_rate']:.2%}, "
        f"side_mae={scores['side_mae']:.2f}, mean_return={scores['mean_return']:.2f}, mean_length={scores['mean_length']:.1f}"
    )


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

    # heightmap: same shuffle-and-holdout as pqn.py, so a shared seed tests the same maps
    records = load_maps(args.maps_folder)
    train_records, test_records = split_maps(records, args.seed, args.eval_fraction)
    test_ids = ", ".join(str(record["id"]) for record in test_records)
    print(f"{len(train_records)} train maps, {len(test_records)} test maps: {test_ids}")

    # env setup
    # heightmap: SAME_STEP reset matches the older gym behavior this file was written for
    envs = gym.vector.SyncVectorEnv(
        [make_env(train_records) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # heightmap: remember which edits were legal so the update uses the same mask
    action_masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_action_mask = torch.tensor(infos["action_mask"]).to(device)  # heightmap
    # heightmap: test-map bookkeeping
    best_test_success = -1.0
    last_eval_step = 0

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
            action_masks[step] = next_action_mask  # heightmap

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=next_action_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_action_mask = torch.tensor(infos["action_mask"]).to(device)  # heightmap

            # heightmap: newer gymnasium groups finished episodes into arrays with a "_" mask of which envs finished
            if "final_info" in infos:
                final_info = infos["final_info"]
                for env_index in np.nonzero(infos["_final_info"])[0]:
                    episode_return = final_info["episode"]["r"][env_index]
                    episode_length = final_info["episode"]["l"][env_index]
                    print(f"global_step={global_step}, episodic_return={episode_return}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar("charts/success", float(final_info["success"][env_index]), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, envs.single_action_space.n))  # heightmap

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds], b_action_masks[mb_inds]  # heightmap: same mask as the rollout
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # heightmap: try the test maps now and then, keep the model that solves the most
        if global_step - last_eval_step >= args.eval_every or iteration == args.num_iterations:
            last_eval_step = global_step
            results = evaluate_on_test_maps(agent, test_records, device)
            scores = score_eval(results)
            for name, value in scores.items():
                writer.add_scalar(f"eval/{name}", value, global_step)
            writer.add_scalar("charts/test_success", scores["success_rate"], global_step)
            print(f"global_step={global_step}, {format_eval(results, scores)}")
            if scores["success_rate"] > best_test_success:
                best_test_success = scores["success_rate"]
                torch.save(agent.state_dict(), f"runs/{run_name}/agent.pt")
                print(f"saved runs/{run_name}/agent.pt")

    envs.close()
    writer.close()
