# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import trange
from torch.autograd import Variable
import itertools



@dataclass
class TrainConfig:
    project: str = ""
    group: str = ""
    name: str = ""
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    env_name: str = "walker2d-medium-expert-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    target_returns: Tuple[float, ...] = (5000.0,)
    eval_episodes: int = 100
    eval_every: int = 10_000
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"
    subgoal_interval: int = 5

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{self.target_returns}--{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
def set_seed(
        seed: int,
        env: Optional[gym.Env] = None,
        deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
        arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def compute_subgoals(rewards, config=TrainConfig):
    rewards = rewards.flatten()
    T = len(rewards)
    subgoal_indices = []

    i = 0

    while i < T - 1:
        best_subgoal_val = float("-inf")
        best_subgoal_idx = i + 1

        for j in range(i + 1, T):
            accumulated_reward = np.sum(rewards[i + 1: j + 1])
            temporal_distance = j - i
            subgoal_value = accumulated_reward / temporal_distance
            if subgoal_value > best_subgoal_val:
                best_subgoal_val = subgoal_value
                best_subgoal_idx = j

        subgoal_indices.extend([best_subgoal_idx] * (best_subgoal_idx - i))
        i = best_subgoal_idx

    # Ensure the last state is a subgoal
    if subgoal_indices != []:
        subgoal_indices.append(T - 1)
    return torch.as_tensor(subgoal_indices, dtype=torch.long)



def load_d4rl_trajectories(
        env_name: str,
        gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):

        data_["observations"].append(dataset["observations"][i])    #(state_dim,)
        data_["actions"].append(dataset["actions"][i])              #(act_dim,)
        data_["rewards"].append(dataset["rewards"][i])              #()

        if (
                dataset["terminals"][i]
                or dataset["timeouts"][i]):
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            sg_indices = compute_subgoals(episode_data['rewards'])
            episode_data['indices'] = sg_indices                    #([episode_len])
            episode_data["subgoals"] = episode_data["observations"][sg_indices]

            episode_data["returns"] = discounted_cumsum(            # (episode_len,)
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])

            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info

#
# class SequenceDataset(IterableDataset):
#     def __init__(
#             self,
#             env_name: str,
#             seq_len: int = 20,
#             reward_scale: float = 1.0
#     ):
#         self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
#         print(f"\n\n {len(info['traj_lens'])}\n\n")
#         self.reward_scale = reward_scale
#         self.seq_len = seq_len
#
#         self.state_mean = info["obs_mean"]
#         self.state_std = info["obs_std"]
#         self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
#
#     def __prepare_sample(self, traj_idx, start_idx):
#         traj = self.dataset[traj_idx]
#
#         states = traj["observations"][start_idx: start_idx + self.seq_len]      # (seq_len, state_dim)
#         actions = traj["actions"][start_idx: start_idx + self.seq_len]          # (seq_len, act_dim)
#         returns = traj["returns"][start_idx: start_idx + self.seq_len]          # (seq_len,)
#         time_steps = np.arange(start_idx, start_idx + self.seq_len)             # (seq_len,)
#         rewards = traj["rewards"][start_idx: start_idx + self.seq_len]          # (seq_len,)
#         sg_states = traj['subgoals'][start_idx: start_idx + self.seq_len]       # (seq_len, state_dim)
#
#         states = (states - self.state_mean) / self.state_std
#         sg_states = (sg_states - self.state_mean) / self.state_std
#
#         # Required if the current sampled trajectory is not equal to seq_len (i.e. its smaller than seq_len)
#         mask = np.hstack([                                                      # (seq_len,)
#             np.ones(states.shape[0]),
#             np.zeros(self.seq_len - states.shape[0])
#         ])
#
#         if states.shape[0] < self.seq_len:
#             states = pad_along_axis(states, pad_to=self.seq_len)
#             sg_states = pad_along_axis(sg_states, pad_to=self.seq_len)
#             actions = pad_along_axis(actions, pad_to=self.seq_len)
#             returns = pad_along_axis(returns, pad_to=self.seq_len)
#             rewards = pad_along_axis(rewards, pad_to=self.seq_len)
#
#         return states, actions, returns, rewards, sg_states, time_steps, mask
#
#     def __iter__(self):
#         while True:
#             traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
#             start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
#             yield self.__prepare_sample(traj_idx, start_idx)


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):

        traj = self.dataset[traj_idx]
        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        rewards = traj["rewards"][start_idx: start_idx + self.seq_len]

        interval = 5
        repeat_length = 5

        selected_values = states[::interval]
        sg_states = []
        countdown_array = []

        for value in selected_values:
            actual_repeats = min(repeat_length, states.shape[0] - len(sg_states) + repeat_length)
            sg_states.extend([value] * actual_repeats)
            countdown_array.extend(range(repeat_length, repeat_length - actual_repeats, -1))

        sg_states = np.array(sg_states)
        countdown_array = np.array(countdown_array, dtype=returns.dtype)

        states = (states - self.state_mean) / self.state_std
        sg_states = (sg_states - self.state_mean) / self.state_std

        # distance_vectors = []
        # for i in range(len(states)):
        #     distance_vectors.append(np.linalg.norm(states[i] - sg_states[i]))
        #
        # distance_vectors = np.array(distance_vectors)

        rewards = rewards * self.reward_scale
        returns = returns * self.reward_scale
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )

        # Pad to sequence length if needed
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            sg_states = pad_along_axis(sg_states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            rewards = pad_along_axis(rewards, pad_to=self.seq_len)
            countdown_array = pad_along_axis(countdown_array, pad_to=self.seq_len)

        return states, actions, returns, rewards, sg_states, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


class GoalTransformerBlock(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(
            self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]

        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x



class GoalTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        self.reward_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.subgoal_head = nn.Sequential(
            nn.Linear(embedding_dim, 2 * state_dim), nn.Tanh()
        )

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.episode_len = episode_len

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            rewards: torch.Tensor,
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        rewards_emb = self.reward_emb(rewards.unsqueeze(-1)) + time_emb

        sequence = (
            torch.stack([returns_emb, state_emb, rewards_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        subgoal_out = self.subgoal_head(out[:, 1::3])

        return subgoal_out


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 5 * embedding_dim),
            nn.GELU(),
            nn.Linear(5 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
            self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.state_dim = state_dim

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.subgoal_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        self.embed_g_to_z = torch.nn.Linear(self.state_dim, self.state_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=4 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, action_dim), nn.Tanh()
        )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            subgoals: torch.Tensor,  # [batch_size, seq_len, state_dim]
            rewards: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        subgoal_emb = self.subgoal_emb(subgoals) + time_emb
        returns_emb = self.return_emb(rewards.unsqueeze(-1)) + time_emb

        sequence = (
            torch.stack([subgoal_emb, state_emb, act_emb, returns_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 4 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 4 * seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        action_out = self.action_head(out[:, 1::4])

        return action_out


    def map_g_to_z(self, h_g):
        z = self.embed_g_to_z(h_g)
        return z


    def get_predict_action(self, states, actions, sg_dist, rewards, timesteps, goal, padding_mask=None):
        if goal is None:
            mu = sg_dist[:, :, :self.state_dim]
            logvar = sg_dist[:, :, self.state_dim:]
            z = self.reparametrize(mu, logvar)
        else:
            z = goal

        action_out = self.forward(states, actions, z, rewards, timesteps, padding_mask)

        return action_out


    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps



def is_subgoal_reached(state: np.ndarray, subgoal: np.ndarray, threshold: float = 1.0) -> bool:

    # Calculate the Euclidean distance between the current state and the subgoal
    distance = np.linalg.norm(state - subgoal)

    # Check if the distance is within the threshold
    return distance, distance <= threshold



@torch.no_grad()
def eval_rollout(
        model: DecisionTransformer,
        goal_model: GoalTransformer,
        env: gym.Env,
        target_return: float,
        device: str = "cpu",
        config=TrainConfig,
) -> Tuple[float, float]:
    states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=device)
    sg_states = torch.zeros(1, model.episode_len + 1, 2 * model.state_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    rewards = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0

    while episode_len < model.episode_len:
        if episode_len >= model.episode_len:
            break

        current_subgoal = goal_model(
            states[:, : episode_len + 1][:, -model.seq_len:],
            returns[:, : episode_len + 1][:, -model.seq_len:],
            rewards[:, : episode_len + 1][:, -model.seq_len:],
            time_steps[:, : episode_len + 1][:, -model.seq_len:],
        )[0, -1].cpu().numpy()

        current_subgoal = current_subgoal.reshape(1, 2 * model.state_dim)
        sg_states[:, episode_len: episode_len + config.subgoal_interval] = torch.as_tensor(
            current_subgoal).repeat(sg_states[:, episode_len: episode_len + config.subgoal_interval].shape[1], 1).to(device)

        subgoal_achieved = 0

        for step in range(config.subgoal_interval):
            predicted_actions = model.get_predict_action(
                states[:, : step + 1][:, -model.seq_len:],
                actions[:, : step + 1][:, -model.seq_len:],
                sg_states[:, : step + 1][:, -model.seq_len:],
                rewards[:, : step + 1][:, -model.seq_len:],
                time_steps[:, : step + 1][:, -model.seq_len:],
                goal=None)

            predicted_action = predicted_actions[0, -1].cpu().numpy()
            next_state, reward, done, _ = env.step(predicted_action)

            actions[:, step] = torch.as_tensor(predicted_action, device=device)
            rewards[:, step] = torch.as_tensor(reward, device=device)
            states[:, step + 1] = torch.as_tensor(next_state, device=device)
            returns[:, step + 1] = returns[:, step] - reward

            episode_return += reward
            episode_len += 1

            if done or episode_len >= model.episode_len:
                return episode_return, episode_len

            # dist, state_reached_subgoal = is_subgoal_reached(states[:, step + 1].cpu().numpy(), current_subgoal)
            # subgoal_step_count += 1
            # dists.append(dist)
            # subgoal_achieved += state_reached_subgoal

            # if state_reached_subgoal or subgoal_step_count >= config.subgoal_interval:
            #     # if subgoal_step_count >= config.subgoal_interval:
            #     #     print(dists[-config.subgoal_interval:])
            #     # Predict a new subgoal if the current one is reached or subgoal interval is exceeded
            #     current_subgoal = goal_model(
            #         states[:, : step + 1][:, -model.seq_len:],
            #         rewards[:, : step + 1][:, -model.seq_len:],
            #         sg_states[:, : step + 1][:, -model.seq_len:],
            #         time_steps[:, : step + 1][:, -model.seq_len:]
            #     )[0, -1].cpu().numpy()
            #     subgoal_step_count = 0  # Reset the subgoal step counter

            # sg_states[:, step + 1] = torch.as_tensor(current_subgoal, device=device)

    return episode_return, episode_len



@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    dataset = SequenceDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]

    print(f'\n\n\n {config.subgoal_interval}\n\n\n')

    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    goal_model = GoalTransformer(
        state_dim=config.state_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(config.device)

    optim = torch.optim.AdamW(
        itertools.chain(goal_model.parameters(), model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)

    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, rewards, sg_states, time_steps, mask = [b.to(config.device) for b in batch]

        # print(states.shape)       torch.Size([batch_size, seq_len, state_dim])
        # print(actions.shape)      torch.Size([batch_size, seq_len, act_dim])
        # print(returns.shape)      torch.Size([batch_size, seq_len])
        # print(rewards.shape)      torch.Size([batch_size, seq_len])
        # print(sg_states.shape)    torch.Size([batch_size, seq_len, state_dim])
        # print(time_steps.shape)   torch.Size([batch_size, seq_len])
        # print(mask.shape)         torch.Size([batch_size, seq_len])

        padding_mask = ~mask.to(torch.bool)

        predicted_sub = goal_model(             # [batch_size, seq_len, 2 * state_dim]
            states=states,
            returns_to_go=returns,
            rewards=rewards,
            time_steps=time_steps,
        )

        predicted_actions_pred_sub = model.get_predict_action(
            states=states,
            actions=actions,
            sg_dist=predicted_sub,
            rewards=returns,
            timesteps=time_steps,
            goal=None,
            padding_mask=padding_mask,
        )
        pred_sub_action_loss = F.mse_loss(predicted_actions_pred_sub, actions.detach(),
                                          reduction="none") * mask.unsqueeze(-1)

        # sg_target = torch.clone(sg_states)
        sg_target = model.map_g_to_z(sg_states)

        predicted_actions_actual_sub = model.get_predict_action(
            states=states,
            actions=actions,
            sg_dist=None,
            rewards=returns,
            timesteps=time_steps,
            goal=sg_target,
            padding_mask=padding_mask,
        )

        actual_sub_action_loss = F.mse_loss(predicted_actions_actual_sub, actions.detach(),
                                            reduction="none") * mask.unsqueeze(-1)


        loss = actual_sub_action_loss.mean() + (0.5) * pred_sub_action_loss.mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        if (step % config.eval_every == 0 or step == config.update_steps - 1) and step != 0:
            model.eval()
            goal_model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                eval_lens = []
                sgs = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        goal_model=goal_model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)
                    eval_lens.append(eval_len)

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )

                print(f"eval/{target_return}_return_mean: ", np.mean(eval_returns))
                print(f"eval/{target_return}_return_std: ", np.std(eval_returns))
                print(f"eval/{target_return}_normalized_score_mean: ", np.mean(normalized_scores))
                print(f"eval/{target_return}_normalized_score_std: ", np.std(normalized_scores))
                print(f"eval/{target_return}_avg_episode_len: ", np.mean(eval_lens))
                # print(f"Action loss: {np.mean(loss)}")
                # print(f"Subgoal Distance loss: {np.mean(sg_dist_loss)}")
                # print(f'Avg Subgoals achieved: {np.mean(sgs)}')
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()
            goal_model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
