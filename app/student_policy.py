"""Lightweight student (v004): the policy DAgger distills and PPO fine-tunes.

A 2x256 MLP trunk on the 47-D observation with two heads:
  * skill head  - 8-way classifier over the assembly skills (labels: DiffusionGemma's decisions)
  * action head - 5-D Gaussian mean (tanh) conditioned on the skill one-hot (labels: the privileged teacher's action)
plus a log-std vector for PPO and a value head. The value head reads DETACHED trunk features: PPO's return-scale
critic loss must never rewrite the features the distilled skill/action heads depend on. ~150 k parameters, ~0.1 ms per step on one CPU core - small enough
for a 20 Hz controller on the edge, where the planner and the 26 B decision model are not.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

N_SKILLS = 8


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.register_buffer("obs_mean", torch.zeros(obs_dim)); self.register_buffer("obs_std", torch.ones(obs_dim))
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.skill_head = nn.Linear(hidden, N_SKILLS)
        self.act_head = nn.Sequential(nn.Linear(hidden + N_SKILLS, hidden), nn.ELU(), nn.Linear(hidden, act_dim))
        self.value_head = nn.Sequential(nn.Linear(hidden + N_SKILLS, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))

    def set_norm(self, obs: np.ndarray) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs.mean(0), dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(obs.std(0) + 1e-3, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, skill: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        h = self.trunk((obs - self.obs_mean) / self.obs_std)
        logits = self.skill_head(h)
        onehot = nn.functional.one_hot(logits.argmax(-1) if skill is None else skill, N_SKILLS).float()
        z = torch.cat([h, onehot], -1)
        return {"skill_logits": logits, "mean": torch.tanh(self.act_head(z)), "value": self.value_head(z.detach()).squeeze(-1)}

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, int]:
        out = self.forward(torch.as_tensor(obs, dtype=torch.float32)[None])
        mean = out["mean"][0]
        a = mean if deterministic else mean + torch.exp(self.log_std) * torch.randn_like(mean)
        return a.clamp(-1, 1).numpy(), int(out["skill_logits"][0].argmax())

    def save(self, path: str) -> None:
        torch.save({"state": self.state_dict(), "obs_dim": self.obs_dim, "act_dim": self.act_dim}, path)

    @staticmethod
    def load(path: str) -> "StudentPolicy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = StudentPolicy(ck["obs_dim"], ck["act_dim"]); p.load_state_dict(ck["state"]); p.eval(); return p
