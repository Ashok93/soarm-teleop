"""Simple episode recorder for pick-and-place demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeRecorder:
    """Accumulate steps and write one NPZ per episode."""

    output_dir: Path
    episode_index: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    language: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_step(self, observation: dict[str, Any], action: np.ndarray, command: str) -> None:
        self.observations.append(observation)
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.language.append(command)

    def reset_episode(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.language.clear()

    def flush(self) -> Path:
        if not self.observations:
            raise RuntimeError("Cannot flush an empty episode.")
        episode_path = self.output_dir / f"episode_{self.episode_index:04d}.npz"

        obs_keys = self.observations[0].keys()
        stacked_obs = {key: np.stack([obs[key] for obs in self.observations]) for key in obs_keys}
        actions = np.stack(self.actions)
        language = np.asarray(self.language)

        np.savez_compressed(episode_path, observations=stacked_obs, actions=actions, language=language)
        self.episode_index += 1
        self.reset_episode()
        return episode_path
