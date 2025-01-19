# # inspiration:
# # 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# # 2. https://github.com/karpathy/minGPT
# import os
# import random
# import uuid
# from collections import defaultdict
# from dataclasses import asdict, dataclass
# from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
#
# import d4rl  # noqa
# import gym
# import numpy as np
# import pyrallis
# import torch
# import torch.nn as nn
# import wandb
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, IterableDataset
# from tqdm.auto import trange
#
#
# @dataclass
# class TrainConfig:
#     project: str = ""
#     group: str = ""
#     name: str = ""
#     embedding_dim: int = 128
#     num_layers: int = 3
#     num_heads: int = 1
#     seq_len: int = 20
#     episode_len: int = 1000
#     attention_dropout: float = 0.1
#     residual_dropout: float = 0.1
#     embedding_dropout: float = 0.1
#     max_action: float = 1.0
#     env_name: str = "walker2d-medium-replay-v2"
#     learning_rate: float = 1e-4
#     betas: Tuple[float, float] = (0.9, 0.999)
#     weight_decay: float = 1e-4
#     clip_grad: Optional[float] = 0.25
#     batch_size: int = 64
#     update_steps: int = 100_000
#     warmup_steps: int = 10_000
#     reward_scale: float = 0.001
#     num_workers: int = 4
#     target_returns: Tuple[float, ...] = (5000.0, )
#     eval_episodes: int = 100
#     eval_every: int = 10_000
#     checkpoints_path: Optional[str] = None
#     deterministic_torch: bool = False
#     train_seed: int = 10
#     eval_seed: int = 42
#     device: str = "cuda"
#     subgoal_interval: int = 5
#
#     def __post_init__(self):
#         self.name = f"{self.name}-{self.env_name}-{self.target_returns}--{str(uuid.uuid4())[:8]}"
#         if self.checkpoints_path is not None:
#             self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
#
#
# # general utils
# def set_seed(
#         seed: int,
#         env: Optional[gym.Env] = None,
#         deterministic_torch: bool = False
# ):
#     if env is not None:
#         env.seed(seed)
#         env.action_space.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.use_deterministic_algorithms(deterministic_torch)
#
#
# def wandb_init(config: dict) -> None:
#     wandb.init(
#         config=config,
#         project=config["project"],
#         group=config["group"],
#         name=config["name"],
#         id=str(uuid.uuid4()),
#     )
#     wandb.run.save()
#
#
# def wrap_env(
#         env: gym.Env,
#         state_mean: Union[np.ndarray, float] = 0.0,
#         state_std: Union[np.ndarray, float] = 1.0,
#         reward_scale: float = 1.0,
# ) -> gym.Env:
#     def normalize_state(state):
#         return (state - state_mean) / state_std
#
#     def scale_reward(reward):
#         return reward_scale * reward
#
#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env
#
#
# # some utils functionalities specific for Decision Transformer
# def pad_along_axis(
#         arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
# ) -> np.ndarray:
#     pad_size = pad_to - arr.shape[axis]
#     if pad_size <= 0:
#         return arr
#
#     npad = [(0, 0)] * arr.ndim
#     npad[axis] = (0, pad_size)
#     return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)
#
#
# def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
#     cumsum = np.zeros_like(x)
#     cumsum[-1] = x[-1]
#     for t in reversed(range(x.shape[0] - 1)):
#         cumsum[t] = x[t] + gamma * cumsum[t + 1]
#     return cumsum
#
#
# def compute_subgoals(rewards, config=TrainConfig):
#     rewards = rewards.flatten()
#     T = len(rewards)
#     subgoal_indices = []
#
#     i = 0
#
#     while i < T - 1:
#         best_subgoal_val = float("-inf")
#         best_subgoal_idx = i + 1
#
#         for j in range(i + 1, min(i+1+config.subgoal_interval, T)):
#             accumulated_reward = np.sum(rewards[i + 1: j + 1])
#             temporal_distance = j - i
#             subgoal_value = accumulated_reward / temporal_distance
#             if subgoal_value > best_subgoal_val:
#                 best_subgoal_val = subgoal_value
#                 best_subgoal_idx = j
#
#         subgoal_indices.extend([best_subgoal_idx] * (best_subgoal_idx - i))
#         i = best_subgoal_idx
#
#     # Ensure the last state is a subgoal
#     if subgoal_indices != []:
#         subgoal_indices.append(T - 1)
#     return torch.as_tensor(subgoal_indices, dtype=torch.long)
#
#
#
# def load_d4rl_trajectories(
#         env_name: str,
#         gamma: float = 1.0
# ) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
#     dataset = gym.make(env_name).get_dataset()
#     traj, traj_len = [], []
#
#     data_ = defaultdict(list)
#     for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
#         data_["observations"].append(dataset["observations"][i])
#         data_["actions"].append(dataset["actions"][i])
#         data_["rewards"].append(dataset["rewards"][i])
#
#         if (
#                 dataset["terminals"][i]
#                 or dataset["timeouts"][i]):
#             episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
#             sg_indices = compute_subgoals(episode_data['rewards'])
#             episode_data['indices'] = sg_indices
#             episode_data["subgoals"] = episode_data["observations"][sg_indices]
#             # return-to-go if gamma=1.0, just discounted returns else
#             episode_data["returns"] = discounted_cumsum(
#                 episode_data["rewards"], gamma=gamma
#             )
#             traj.append(episode_data)
#             traj_len.append(episode_data["actions"].shape[0])
#             # reset trajectory buffer
#             data_ = defaultdict(list)
#
#     # needed for normalization, weighted sampling, other stats can be added also
#     info = {
#         "obs_mean": dataset["observations"].mean(0, keepdims=True),
#         "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
#         "traj_lens": np.array(traj_len),
#     }
#     return traj, info
#
#
# def desired_states(s, r):
#     max_norm_return = 0
#     accumlated_return = 0
#     max_idx = 0
#     for t in range(0, r.shape[0]):
#         accumlated_return += r[t]
#         norm_return = accumlated_return / (t + 1)
#         if norm_return > max_norm_return:
#             max_norm_return = norm_return
#             max_idx = t
#     tmp = min(max_idx+1, r.shape[0]-1)
#     goal = s[tmp]
#     return goal
#
#
# class SequenceDataset(IterableDataset):
#     def __init__(
#             self,
#             env_name: str,
#             seq_len: int = 10,
#             reward_scale: float = 1.0
#     ):
#         self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
#         print(f"\n\n {len(info['traj_lens'])}\n\n")
#         self.reward_scale = reward_scale
#         self.seq_len = seq_len
#
#         self.state_mean = info["obs_mean"]
#         self.state_std = info["obs_std"]
#         # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
#         self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
#
#     def __prepare_sample(self, traj_idx, start_idx):
#         traj = self.dataset[traj_idx]
#
#         states = traj["observations"][start_idx: start_idx + self.seq_len]
#         actions = traj["actions"][start_idx: start_idx + self.seq_len]
#         returns = traj["returns"][start_idx: start_idx + self.seq_len]
#         time_steps = np.arange(start_idx, start_idx + self.seq_len)
#         rewards = traj["rewards"][start_idx: start_idx + self.seq_len]
#
#         sg_state = desired_states(states, rewards)
#         sg_states = torch.as_tensor(sg_state.reshape(1, states.shape[1])).repeat(states.shape[0], 1)
#
#         # Normalize states, subgoal states, and distance vectors
#         states = (states - self.state_mean) / self.state_std
#         sg_states = (sg_states - self.state_mean) / self.state_std
#         distance_vectors = torch.as_tensor(torch.as_tensor(states) - sg_states)
#
#         # Scale rewards and prepare mask for padding
#         returns = returns * self.reward_scale
#         mask = np.hstack(
#             [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
#         )
#
#         # Pad to sequence length if needed
#         if states.shape[0] < self.seq_len:
#             states = pad_along_axis(states, pad_to=self.seq_len)
#             sg_states = pad_along_axis(sg_states, pad_to=self.seq_len)
#             actions = pad_along_axis(actions, pad_to=self.seq_len)
#             returns = pad_along_axis(returns, pad_to=self.seq_len)
#             distance_vectors = pad_along_axis(distance_vectors, pad_to=self.seq_len)
#
#         return states, actions, returns, sg_states, distance_vectors, time_steps, mask
#
#     def __iter__(self):
#         while True:
#             traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
#             start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
#             yield self.__prepare_sample(traj_idx, start_idx)
#
#
#
# class GoalTransformerBlock(nn.Module):
#     def __init__(
#             self,
#             seq_len: int,
#             embedding_dim: int,
#             num_heads: int,
#             attention_dropout: float,
#             residual_dropout: float,
#     ):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.drop = nn.Dropout(residual_dropout)
#
#         self.attention = nn.MultiheadAttention(
#             embedding_dim, num_heads, attention_dropout, batch_first=True
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 4 * embedding_dim),
#             nn.GELU(),
#             nn.Linear(4 * embedding_dim, embedding_dim),
#             nn.Dropout(residual_dropout),
#         )
#         # True value indicates that the corresponding position is not allowed to attend
#         self.register_buffer(
#             "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
#         )
#         self.seq_len = seq_len
#
#     def forward(
#             self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
#
#         norm_x = self.norm1(x)
#         attention_out = self.attention(
#             query=norm_x,
#             key=norm_x,
#             value=norm_x,
#             attn_mask=causal_mask,
#             key_padding_mask=padding_mask,
#             need_weights=False,
#         )[0]
#
#         x = x + self.drop(attention_out)
#         x = x + self.mlp(self.norm2(x))
#         return x
#
#
# class GoalTransformer(nn.Module):
#     def __init__(
#             self,
#             state_dim: int,
#             seq_len: int = 10,
#             episode_len: int = 1000,
#             embedding_dim: int = 128,
#             num_layers: int = 4,
#             num_heads: int = 8,
#             attention_dropout: float = 0.0,
#             residual_dropout: float = 0.0,
#             embedding_dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.emb_drop = nn.Dropout(embedding_dropout)
#         self.emb_norm = nn.LayerNorm(embedding_dim)
#
#         self.out_norm = nn.LayerNorm(embedding_dim)
#         self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
#         self.state_emb = nn.Linear(state_dim, embedding_dim)
#         self.return_emb = nn.Linear(1, embedding_dim)
#
#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     seq_len=3 * seq_len,
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     attention_dropout=attention_dropout,
#                     residual_dropout=residual_dropout,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#
#         self.subgoal_head = nn.Sequential(
#             nn.Linear(embedding_dim, state_dim), nn.Tanh()
#         )
#
#         self.seq_len = seq_len
#         self.embedding_dim = embedding_dim
#         self.state_dim = state_dim
#         self.episode_len = episode_len
#
#         self.apply(self._init_weights)
#
#     @staticmethod
#     def _init_weights(module: nn.Module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)
#
#     def forward(
#             self,
#             states: torch.Tensor,  # [batch_size, seq_len, state_dim]
#             returns_to_go: torch.Tensor,  # [batch_size, seq_len]
#             subgoals: torch.Tensor,  # [batch_size, seq_len, state_dim]
#             time_steps: torch.Tensor,  # [batch_size, seq_len]
#             padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
#     ) -> torch.FloatTensor:
#         batch_size, seq_len = states.shape[0], states.shape[1]
#         time_emb = self.timestep_emb(time_steps)
#         state_emb = self.state_emb(states) + time_emb
#         returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
#         subgoal_emb = self.state_emb(subgoals) + time_emb
#
#         sequence = (
#             torch.stack([returns_emb, state_emb, subgoal_emb], dim=1)
#             .permute(0, 2, 1, 3)
#             .reshape(batch_size, 3 * seq_len, self.embedding_dim)
#         )
#
#         if padding_mask is not None:
#             padding_mask = (
#                 torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
#                 .permute(0, 2, 1)
#                 .reshape(batch_size, 3 * seq_len)
#             )
#
#         out = self.emb_norm(sequence)
#         out = self.emb_drop(out)
#
#         for block in self.blocks:
#             out = block(out, padding_mask=padding_mask)
#
#         out = self.out_norm(out)
#         subgoal_out = self.subgoal_head(out[:, 1::3])
#
#         return subgoal_out
#
#
#
#
# # Decision Transformer implementation
# class TransformerBlock(nn.Module):
#     def __init__(
#             self,
#             seq_len: int,
#             embedding_dim: int,
#             num_heads: int,
#             attention_dropout: float,
#             residual_dropout: float,
#     ):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.drop = nn.Dropout(residual_dropout)
#
#         self.attention = nn.MultiheadAttention(
#             embedding_dim, num_heads, attention_dropout, batch_first=True
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 6 * embedding_dim),
#             nn.GELU(),
#             nn.Linear(6 * embedding_dim, embedding_dim),
#             nn.Dropout(residual_dropout),
#         )
#         # True value indicates that the corresponding position is not allowed to attend
#         self.register_buffer(
#             "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
#         )
#         self.seq_len = seq_len
#
#     # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
#     def forward(
#             self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
#
#         norm_x = self.norm1(x)
#         attention_out = self.attention(
#             query=norm_x,
#             key=norm_x,
#             value=norm_x,
#             attn_mask=causal_mask,
#             key_padding_mask=padding_mask,
#             need_weights=False,
#         )[0]
#         # by default pytorch attention does not use dropout
#         # after final attention weights projection, while minGPT does:
#         # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
#         x = x + self.drop(attention_out)
#         x = x + self.mlp(self.norm2(x))
#         return x
#
#
# class DecisionTransformer(nn.Module):
#     def __init__(
#             self,
#             state_dim: int,
#             action_dim: int,
#             seq_len: int = 10,
#             episode_len: int = 1000,
#             embedding_dim: int = 128,
#             num_layers: int = 4,
#             num_heads: int = 8,
#             attention_dropout: float = 0.0,
#             residual_dropout: float = 0.0,
#             embedding_dropout: float = 0.0,
#             max_action: float = 1.0,
#     ):
#         super().__init__()
#         self.emb_drop = nn.Dropout(embedding_dropout)
#         self.emb_norm = nn.LayerNorm(embedding_dim)
#
#         self.out_norm = nn.LayerNorm(embedding_dim)
#         # additional seq_len embeddings for padding timesteps
#         self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
#         self.state_emb = nn.Linear(state_dim, embedding_dim)
#         self.action_emb = nn.Linear(action_dim, embedding_dim)
#         self.return_emb = nn.Linear(1, embedding_dim)
#         self.dist_emb = nn.Linear(state_dim, embedding_dim)
#
#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     seq_len=4 * seq_len,
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     attention_dropout=attention_dropout,
#                     residual_dropout=residual_dropout,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         self.action_head = nn.Sequential(
#             nn.Linear(embedding_dim, action_dim), nn.Tanh()
#         )
#         self.subgoal_head = nn.Sequential(
#             nn.Linear(embedding_dim, state_dim), nn.Tanh()
#         )
#         self.seq_len = seq_len
#         self.embedding_dim = embedding_dim
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.episode_len = episode_len
#         self.max_action = max_action
#
#         self.apply(self._init_weights)
#
#     @staticmethod
#     def _init_weights(module: nn.Module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)
#
#     def forward(
#             self,
#             states: torch.Tensor,  # [batch_size, seq_len, state_dim]
#             actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
#             returns_to_go: torch.Tensor,  # [batch_size, seq_len]
#             subgoals: torch.Tensor,  # [batch_size, seq_len, state_dim]
#             distances: torch.Tensor,
#             time_steps: torch.Tensor,  # [batch_size, seq_len]
#             padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
#     ) -> torch.FloatTensor:
#         batch_size, seq_len = states.shape[0], states.shape[1]
#         time_emb = self.timestep_emb(time_steps)
#         state_emb = self.state_emb(states) + time_emb
#         act_emb = self.action_emb(actions) + time_emb
#         subgoal_emb = self.state_emb(subgoals) + time_emb
#         dist_emb = self.dist_emb(distances) + time_emb
#
#         # print(self.state_emb.shape)
#         returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
#
#         # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
#         sequence = (
#             torch.stack([returns_emb, state_emb, act_emb, subgoal_emb, dist_emb], dim=1)
#             .permute(0, 2, 1, 3)
#             .reshape(batch_size, 5 * seq_len, self.embedding_dim)
#         )
#         if padding_mask is not None:
#             # [batch_size, seq_len * 3], stack mask identically to fit the sequence
#             padding_mask = (
#                 torch.stack([padding_mask, padding_mask, padding_mask, padding_mask, padding_mask], dim=1)
#                 .permute(0, 2, 1)
#                 .reshape(batch_size, 5 * seq_len)
#             )
#         # LayerNorm and Dropout (!!!) as in original implementation,
#         # while minGPT & huggingface uses only embedding dropout
#         out = self.emb_norm(sequence)
#         out = self.emb_drop(out)
#
#         for block in self.blocks:
#             out = block(out, padding_mask=padding_mask)
#
#         out = self.out_norm(out)
#         # [batch_size, seq_len, action_dim]
#         # predict actions only from state embeddings
#         # action_out = self.action_head(torch.cat((out[:, 1::4], out[:, 3::4]), dim=-1)) * self.max_action
#
#         action_out = self.action_head(out[:, 1::4])# + out[:, 3::4])
#         subgoal_out = self.subgoal_head(torch.cat((out[:, 0::4], out[:, 1::4]), dim=-1))
#
#
#         out = (action_out, subgoal_out)
#         return out
#
#
#
# def is_subgoal_reached(state: np.ndarray, subgoal: np.ndarray, threshold: float = 1.0) -> bool:
#     distance = np.linalg.norm(state - subgoal)
#     return distance, distance <= threshold
#
#
#
# @torch.no_grad()
# def eval_rollout(
#         model: DecisionTransformer,
#         goal_model: GoalTransformer,
#         env: gym.Env,
#         target_return: float,
#         device: str = "cpu",
#         config=TrainConfig,
# ) -> Tuple[float, float]:
#     states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
#     actions = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=device)
#     sg_states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
#     distances = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
#     returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
#     rewards = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
#     time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)
#
#     states[:, 0] = torch.as_tensor(env.reset(), device=device)
#     returns[:, 0] = torch.as_tensor(target_return, device=device)
#
#     episode_return, episode_len = 0.0, 0
#     subgoal_step_count = 0
#
#     # Step 0: Predict initial subgoal based on state and return-to-go
#     current_subgoal = goal_model(
#         states[:, :1],
#         returns[:, :1],
#         sg_states[:, :1],
#         time_steps[:, :1]
#     )[0, -1].cpu().numpy()
#     subgoal_achieved = 0
#     dists = []
#
#     distances[:, 0] = torch.as_tensor(states[:, 0] - current_subgoal, device=device)
#
#     for step in range(model.episode_len):
#         predicted_actions, _ = model(
#             states[:, : step + 1][:, -model.seq_len:],
#             actions[:, : step + 1][:, -model.seq_len:],
#             returns[:, : step + 1][:, -model.seq_len:],
#             sg_states[:, : step + 1][:, -model.seq_len:],
#             distances[:, : step + 1][:, -model.seq_len:],
#             time_steps[:, : step + 1][:, -model.seq_len:]
#         )
#
#         predicted_action = predicted_actions[0, -1].cpu().numpy()
#         next_state, reward, done, _ = env.step(predicted_action)
#
#         # Update the trajectory buffers
#         actions[:, step] = torch.as_tensor(predicted_action, device=device)
#         rewards[:, step] = torch.as_tensor(reward, device=device)
#         states[:, step + 1] = torch.as_tensor(next_state, device=device)
#         returns[:, step + 1] = returns[:, step] - reward
#         distances[:, step+1] = torch.as_tensor(next_state - current_subgoal, device=device)
#
#         # Check if the subgoal has been reached
#         dist, state_reached_subgoal = is_subgoal_reached(states[:, step + 1].cpu().numpy(), current_subgoal)
#         subgoal_step_count += 1
#         dists.append(dist)
#         subgoal_achieved += state_reached_subgoal
#
#         if state_reached_subgoal or subgoal_step_count >= config.subgoal_interval:
#             # if subgoal_step_count >= config.subgoal_interval:
#             #     print(dists[-config.subgoal_interval:])
#             # Predict a new subgoal if the current one is reached or subgoal interval is exceeded
#             current_subgoal = goal_model(
#                 states[:, : step + 1][:, -model.seq_len:],
#                 returns[:, : step + 1][:, -model.seq_len:],
#                 sg_states[:, : step + 1][:, -model.seq_len:],
#                 time_steps[:, : step + 1][:, -model.seq_len:]
#             )[0, -1].cpu().numpy()
#             subgoal_step_count = 0  # Reset the subgoal step counter
#
#         # Set the subgoal in the sg_states buffer
#         sg_states[:, step + 1] = torch.as_tensor(current_subgoal, device=device)
#
#         episode_return += reward
#         episode_len += 1
#
#         if done:
#             break
#
#     return episode_return, episode_len, subgoal_achieved
#
#
#
# @pyrallis.wrap()
# def train(config: TrainConfig):
#     set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
#     # init wandb session for logging
#     wandb_init(asdict(config))
#     print(f'\n\nEnv: {config.env_name}')
#     print(f'\n\nTarget Returns: {config.target_returns}')
#
#     # data & dataloader setup
#     dataset = SequenceDataset(
#         config.env_name,
#         seq_len=config.seq_len,
#         reward_scale=config.reward_scale
#     )
#     trainloader = DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         pin_memory=True,
#         num_workers=config.num_workers,
#     )
#     # evaluation environment with state & reward preprocessing (as in dataset above)
#     eval_env = wrap_env(
#         env=gym.make(config.env_name),
#         state_mean=dataset.state_mean,
#         state_std=dataset.state_std,
#         reward_scale=config.reward_scale,
#     )
#     # model & optimizer & scheduler setup
#     config.state_dim = eval_env.observation_space.shape[0]
#     config.action_dim = eval_env.action_space.shape[0]
#
#     print(f'\n\n\n {config.subgoal_interval}\n\n\n')
#
#     model = DecisionTransformer(
#         state_dim=config.state_dim,
#         action_dim=config.action_dim,
#         embedding_dim=config.embedding_dim,
#         seq_len=config.seq_len,
#         episode_len=config.episode_len,
#         num_layers=config.num_layers,
#         num_heads=config.num_heads,
#         attention_dropout=config.attention_dropout,
#         residual_dropout=config.residual_dropout,
#         embedding_dropout=config.embedding_dropout,
#         max_action=config.max_action,
#     ).to(config.device)
#
#     goal_model = GoalTransformer(
#         state_dim=config.state_dim,
#         embedding_dim=config.embedding_dim,
#         seq_len=config.seq_len,
#         episode_len=config.episode_len,
#         num_layers=config.num_layers,
#         num_heads=config.num_heads,
#         attention_dropout=config.attention_dropout,
#         residual_dropout=config.residual_dropout,
#         embedding_dropout=config.embedding_dropout,
#     ).to(config.device)
#
#     optim = torch.optim.AdamW(
#         model.parameters(),
#         lr=config.learning_rate,
#         weight_decay=config.weight_decay,
#         betas=config.betas,
#     )
#
#     goal_optim = torch.optim.AdamW(
#         goal_model.parameters(),
#         lr=config.learning_rate,
#         weight_decay=config.weight_decay,
#         betas=config.betas,
#     )
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optim,
#         lambda steps: min((steps + 1) / config.warmup_steps, 1),
#     )
#     # save config to the checkpoint
#     if config.checkpoints_path is not None:
#         print(f"Checkpoints path: {config.checkpoints_path}")
#         os.makedirs(config.checkpoints_path, exist_ok=True)
#         with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
#             pyrallis.dump(config, f)
#
#     print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
#     trainloader_iter = iter(trainloader)
#     sg_loss = []
#     act_loss = []
#     sg_dist_loss = []
#
#     ## Goal Model Training
#
#     for step in trange(config.update_steps, desc="Training"):
#         batch = next(trainloader_iter)
#         states, actions, returns, sg_states, distances, time_steps, mask = [b.to(config.device) for b in batch]
#         # True value indicates that the corresponding key value will be ignored
#         padding_mask = ~mask.to(torch.bool)
#
#         predicted_sub = goal_model(
#             states=states,
#             returns_to_go=returns,
#             subgoals=sg_states,
#             time_steps=time_steps,
#         )
#
#         subgoal_loss = F.mse_loss(predicted_sub, sg_states.detach(), reduction="none")
#         subgoal_loss = subgoal_loss.mean()
#         subgoal_loss = (subgoal_loss * mask.unsqueeze(-1)).mean()
#
#         sg_loss.append(subgoal_loss.cpu().detach().numpy())
#
#         goal_optim.zero_grad()
#         subgoal_loss.backward()
#         if config.clip_grad is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
#         goal_optim.step()
#
#         if step % config.eval_every == 0 or step == config.update_steps - 1:
#             print(f"Subgoal loss: {np.mean(sg_loss)}")
#
#
#
#     for step in trange(config.update_steps, desc="Training"):
#         batch = next(trainloader_iter)
#         states, actions, returns, sg_states, distances, time_steps, mask = [b.to(config.device) for b in batch]
#         padding_mask = ~mask.to(torch.bool)
#
#         predicted_actions, predicted_subgoal = model(
#             states=states,
#             actions=actions,
#             returns_to_go=returns,
#             subgoals=sg_states,
#             distances=distances,
#             time_steps=time_steps,
#             padding_mask=padding_mask,
#         )
#
#         action_loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
#         subgoal_distance_loss = F.mse_loss(states, sg_states.detach(), reduction="none")
#         subgoal_distance_loss = (subgoal_distance_loss * mask.unsqueeze(-1)).mean()
#         action_loss = (action_loss * mask.unsqueeze(-1)).mean()
#         loss = action_loss + subgoal_distance_loss
#
#         act_loss.append(action_loss.cpu().detach().numpy())
#         sg_dist_loss.append(subgoal_distance_loss.cpu().detach().numpy())
#
#         optim.zero_grad()
#         loss.backward()
#         if config.clip_grad is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
#         optim.step()
#         scheduler.step()
#
#         if step % config.eval_every == 0 or step == config.update_steps - 1:
#             model.eval()
#             goal_model.eval()
#             for target_return in config.target_returns:
#                 eval_env.seed(config.eval_seed)
#                 eval_returns = []
#                 eval_lens = []
#                 sgs = []
#                 for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
#                     eval_return, eval_len = eval_rollout(
#                         model=model,
#                         goal_model=goal_model,
#                         env=eval_env,
#                         target_return=target_return * config.reward_scale,
#                         device=config.device,
#                     )
#                     # unscale for logging & correct normalized score computation
#                     eval_returns.append(eval_return / config.reward_scale)
#                     eval_lens.append(eval_len)
#                     sgs.append(sg_achieved)
#
#                 normalized_scores = (
#                         eval_env.get_normalized_score(np.array(eval_returns)) * 100
#                 )
#
#                 print(f"eval/{target_return}_return_mean: ", np.mean(eval_returns))
#                 print(f"eval/{target_return}_return_std: ", np.std(eval_returns))
#                 print(f"eval/{target_return}_normalized_score_mean: ", np.mean(normalized_scores))
#                 print(f"eval/{target_return}_normalized_score_std: ", np.std(normalized_scores))
#                 print(f"eval/{target_return}_avg_episode_len: ", np.mean(eval_lens))
#                 print(f"Action loss: {np.mean(act_loss)}")
#                 print(f"Subgoal Distance loss: {np.mean(sg_dist_loss)}")
#                 print(f'Avg Subgoals achieved: {np.mean(sgs)}')
#                 wandb.log(
#                     {
#                         f"eval/{target_return}_return_mean": np.mean(eval_returns),
#                         f"eval/{target_return}_return_std": np.std(eval_returns),
#                         f"eval/{target_return}_normalized_score_mean": np.mean(
#                             normalized_scores
#                         ),
#                         f"eval/{target_return}_normalized_score_std": np.std(
#                             normalized_scores
#                         ),
#                     },
#                     step=step,
#                 )
#             model.train()
#
#     if config.checkpoints_path is not None:
#         checkpoint = {
#             "model_state": model.state_dict(),
#             "state_mean": dataset.state_mean,
#             "state_std": dataset.state_std,
#         }
#         torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))
#
#
# if __name__ == "__main__":
#     train()



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
    env_name: str = "halfcheetah-medium-replay-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    target_returns: Tuple[float, ...] = (6000.0, )
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

        for j in range(i + 1, min(i+1+config.subgoal_interval, T)):
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
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if (
                dataset["terminals"][i]
                or dataset["timeouts"][i]):
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            sg_indices = compute_subgoals(episode_data['rewards'])
            episode_data['indices'] = sg_indices
            episode_data["subgoals"] = episode_data["observations"][sg_indices]
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(
            self,
            env_name: str,
            seq_len: int = 10,
            reward_scale: float = 1.0
    ):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        print(f"\n\n {len(info['traj_lens'])}\n\n")
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa

        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        rewards = traj["rewards"][start_idx: start_idx + self.seq_len]
        sg_states = traj['subgoals'][start_idx: start_idx + self.seq_len]

        # slices = torch.as_tensor([-1] * len(rewards), dtype=torch.long)
        # sg_states = states[slices]

        states = (states - self.state_mean) / self.state_std
        sg_states = (sg_states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            sg_states = pad_along_axis(sg_states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, sg_states, time_steps, mask

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
            nn.Linear(embedding_dim, state_dim), nn.Tanh()
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
            subgoals: torch.Tensor,  # [batch_size, seq_len, state_dim]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        subgoal_emb = self.state_emb(subgoals) + time_emb

        sequence = (
            torch.stack([returns_emb, state_emb, subgoal_emb], dim=1)
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

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

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
        self.subgoal_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, state_dim), nn.Tanh()
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
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            subgoals: torch.Tensor,  # [batch_size, seq_len, state_dim]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb

        act_emb = self.action_emb(actions) + time_emb
        # print(self.state_emb.shape)
        subgoal_emb = self.state_emb(subgoals) + time_emb

        # print(self.state_emb.shape)
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb, subgoal_emb], dim=1)
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
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        # action_out = self.action_head(torch.cat((out[:, 1::4], out[:, 3::4]), dim=-1)) * self.max_action

        action_out = self.action_head(out[:, 1::4])# + out[:, 3::4])
        subgoal_out = self.subgoal_head(torch.cat((out[:, 0::4], out[:, 1::4]), dim=-1))


        out = (action_out, subgoal_out)
        return out



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
    sg_states = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    rewards = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0
    subgoal_step_count = 0

    # Step 0: Predict initial subgoal based on state and return-to-go
    current_subgoal = goal_model(
        states[:, :1],
        returns[:, :1],
        sg_states[:, :1],
        time_steps[:, :1]
    )[0, -1].cpu().numpy()
    subgoal_achieved = 0
    dists = []
    for step in range(model.episode_len):

        predicted_actions, _ = model(
            states[:, : step + 1][:, -model.seq_len:],
            actions[:, : step + 1][:, -model.seq_len:],
            returns[:, : step + 1][:, -model.seq_len:],
            sg_states[:, : step + 1][:, -model.seq_len:],  # including subgoal states
            time_steps[:, : step + 1][:, -model.seq_len:]
        )

        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, _ = env.step(predicted_action)

        # Update the trajectory buffers
        actions[:, step] = torch.as_tensor(predicted_action, device=device)
        rewards[:, step] = torch.as_tensor(reward, device=device)
        states[:, step + 1] = torch.as_tensor(next_state, device=device)
        returns[:, step + 1] = returns[:, step] - reward

        # Check if the subgoal has been reached
        dist, state_reached_subgoal = is_subgoal_reached(states[:, step + 1].cpu().numpy(), current_subgoal)
        subgoal_step_count += 1
        dists.append(dist)
        subgoal_achieved += state_reached_subgoal

        if state_reached_subgoal or subgoal_step_count >= config.subgoal_interval:
            # if subgoal_step_count >= config.subgoal_interval:
            #     print(dists[-config.subgoal_interval:])
            # Predict a new subgoal if the current one is reached or subgoal interval is exceeded
            current_subgoal = goal_model(
                states[:, : step + 1][:, -model.seq_len:],
                returns[:, : step + 1][:, -model.seq_len:],
                sg_states[:, : step + 1][:, -model.seq_len:],
                time_steps[:, : step + 1][:, -model.seq_len:]
            )[0, -1].cpu().numpy()
            subgoal_step_count = 0  # Reset the subgoal step counter

        # Set the subgoal in the sg_states buffer
        sg_states[:, step + 1] = torch.as_tensor(current_subgoal, device=device)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len, subgoal_achieved



@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))
    print(f'\n\nEnv: {config.env_name}')
    print(f'\n\nTarget Returns: {config.target_returns}')

    # data & dataloader setup
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
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    goal_optim = torch.optim.AdamW(
        goal_model.parameters(),
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
    sg_loss = []
    act_loss = []
    sg_dist_loss = []

    ## Goal Model Training

    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, sg_states, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_sub = goal_model(
            states=states,
            returns_to_go=returns,
            subgoals=sg_states,
            time_steps=time_steps,
        )

        subgoal_loss = F.mse_loss(predicted_sub, sg_states.detach(), reduction="none")
        subgoal_loss = subgoal_loss.mean()
        subgoal_loss = (subgoal_loss * mask.unsqueeze(-1)).mean()

        sg_loss.append(subgoal_loss.cpu().detach().numpy())

        goal_optim.zero_grad()
        subgoal_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        goal_optim.step()

        if step % config.eval_every == 0 or step == config.update_steps - 1:
            print(f"Subgoal loss: {np.mean(sg_loss)}")



    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, sg_states, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions, predicted_subgoal = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            subgoals=sg_states,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        action_loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        subgoal_distance_loss = F.mse_loss(states, sg_states.detach(), reduction="none")
        subgoal_distance_loss = (subgoal_distance_loss * mask.unsqueeze(-1)).mean()
        action_loss = (action_loss * mask.unsqueeze(-1)).mean()
        loss = action_loss + subgoal_distance_loss

        act_loss.append(action_loss.cpu().detach().numpy())
        sg_dist_loss.append(subgoal_distance_loss.cpu().detach().numpy())

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            goal_model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                eval_lens = []
                sgs = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len, sg_achieved = eval_rollout(
                        model=model,
                        goal_model=goal_model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)
                    eval_lens.append(eval_len)
                    sgs.append(sg_achieved)

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )

                print(f"eval/{target_return}_return_mean: ", np.mean(eval_returns))
                print(f"eval/{target_return}_return_std: ", np.std(eval_returns))
                print(f"eval/{target_return}_normalized_score_mean: ", np.mean(normalized_scores))
                print(f"eval/{target_return}_normalized_score_std: ", np.std(normalized_scores))
                print(f"eval/{target_return}_avg_episode_len: ", np.mean(eval_lens))
                print(f"Action loss: {np.mean(act_loss)}")
                print(f"Subgoal Distance loss: {np.mean(sg_dist_loss)}")
                print(f'Avg Subgoals achieved: {np.mean(sgs)}')
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

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
