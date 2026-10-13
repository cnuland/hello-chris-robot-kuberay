"""Lightweight student for the multi-part cell (v006).

  part-ID head : 16x16 height scan -> which part is being handled   (labels: DiffusionGemma VISION on the RGB camera)
  trunk        : proprioception + geometry + flags + part one-hot
  skill head   : 12-way next-skill classifier                        (labels: DiffusionGemma's procedure decisions)
                 v006c: kept as the student's explainable "what I am doing" output; NOT fed to the action head -
                 in the multi-part cell DiffusionGemma's step labels run one phase ahead/behind (weld states read
                 "return_tool" 73% of the time) and conditioning motion on them misled the action head.
  selection    : the part-ID one-hot GATHERS the identified part's nest vector and its tool block (driver / torch)
                 exactly - a concatenating MLP only approximates that multiplicative gate, and the error drifted
                 the 250-step place carry.
  action head  : 4-D motion (tanh) + a separate GRIPPER LOGIT, on [trunk, part one-hot, selected nest/tool] (labels:
                 the privileged teacher). v006b: the binary gripper is a classifier (BCE), not a regressed 5th dimension -
                 a regressed grip flickered across its threshold on ~2% of place steps and every flicker dropped a part.
  value head   : on DETACHED trunk features (PPO; see v005)

~0.9 M parameters (512-wide), sub-millisecond on one CPU core. The scan is the last SCAN_DIM entries of the observation.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

SCAN_DIM = 256
N_SKILLS = 12


class MultiStudent(nn.Module):
    def __init__(self, obs_dim: int, n_parts: int, act_dim: int = 5, hidden: int = 512, tool_map=None):
        super().__init__()
        self.obs_dim, self.n_parts, self.act_dim = obs_dim, n_parts, act_dim
        # obs layout (multi_env._obs): 16 + 3 per part (nests) then 8 per tool; tool_map[p][t] = 1 if part p uses tool t
        self.nest_off, self.tool_off, self.n_tools = 16, 16 + 3 * n_parts, 2
        self.register_buffer("tool_map", torch.as_tensor(tool_map if tool_map is not None else np.zeros((n_parts, 2)), dtype=torch.float32))
        core = obs_dim - SCAN_DIM
        self.register_buffer("obs_mean", torch.zeros(core)); self.register_buffer("obs_std", torch.ones(core))
        self.id_head = nn.Sequential(nn.Linear(SCAN_DIM, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, n_parts))
        self.trunk = nn.Sequential(nn.Linear(core + n_parts, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.skill_head = nn.Linear(hidden, N_SKILLS)
        z = hidden + n_parts + 3 + 8
        self.act_head = nn.Sequential(nn.Linear(z, hidden), nn.ELU(), nn.Linear(hidden, act_dim))   # [:4] motion, [4] grip logit
        self.value_head = nn.Sequential(nn.Linear(z, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))

    def set_norm(self, obs: np.ndarray) -> None:
        c = obs[:, :-SCAN_DIM]
        self.obs_mean.copy_(torch.as_tensor(c.mean(0), dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(c.std(0) + 1e-3, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, skill: Optional[torch.Tensor] = None, part: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        core, scan = obs[:, :-SCAN_DIM], obs[:, -SCAN_DIM:]
        id_logits = self.id_head(scan)
        p1 = nn.functional.one_hot(id_logits.argmax(-1) if part is None else part, self.n_parts).float()
        h = self.trunk(torch.cat([(core - self.obs_mean) / self.obs_std, p1], -1))
        logits = self.skill_head(h)
        nests = obs[:, self.nest_off:self.tool_off].reshape(-1, self.n_parts, 3)
        tools = obs[:, self.tool_off:self.tool_off + 8 * self.n_tools].reshape(-1, self.n_tools, 8)
        nest_sel = (p1[:, :, None] * nests).sum(1) * 10.0                       # exact gather by the predicted part
        tool_sel = ((p1 @ self.tool_map)[:, :, None] * tools).sum(1) * torch.tensor([10.0] * 6 + [1.0, 1.0])
        z = torch.cat([h, p1, nest_sel, tool_sel], -1)
        out = self.act_head(z); grip_logit = out[:, 4]
        mean = torch.cat([torch.tanh(out[:, :4]), torch.where(grip_logit > 0, 1.0, -1.0)[:, None]], -1)
        return {"id_logits": id_logits, "skill_logits": logits, "mean": mean, "motion": torch.tanh(out[:, :4]),
                "grip_logit": grip_logit, "value": self.value_head(z.detach()).squeeze(-1)}

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, int, int]:
        out = self.forward(torch.as_tensor(obs, dtype=torch.float32)[None]); mean = out["mean"][0]
        a = mean if deterministic else mean + torch.exp(self.log_std) * torch.randn_like(mean)
        return a.clamp(-1, 1).numpy(), int(out["skill_logits"][0].argmax()), int(out["id_logits"][0].argmax())

    def blob(self) -> bytes:
        import io
        b = io.BytesIO(); torch.save({"state": self.state_dict(), "obs_dim": self.obs_dim, "n_parts": self.n_parts,
                                      "act_dim": self.act_dim}, b); return b.getvalue()

    def save(self, path: str) -> None:
        open(path, "wb").write(self.blob())

    @staticmethod
    def load(path_or_blob) -> "MultiStudent":
        import io
        src = io.BytesIO(path_or_blob) if isinstance(path_or_blob, (bytes, bytearray)) else path_or_blob
        ck = torch.load(src, map_location="cpu", weights_only=False)
        hidden = ck["state"]["trunk.0.weight"].shape[0]
        p = MultiStudent(ck["obs_dim"], ck["n_parts"], ck["act_dim"], hidden, tool_map=ck["state"]["tool_map"].numpy())
        p.load_state_dict(ck["state"]); p.eval(); return p
