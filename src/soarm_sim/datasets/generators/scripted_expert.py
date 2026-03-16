"""Scripted expert policy for pick-and-place demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertPhase:
    name: str
    target_pos: np.ndarray
    target_quat: np.ndarray
    gripper_closed: bool
    duration_steps: int


class PickPlaceScriptedExpert:
    """Simple time-based scripted expert.

    This is a deterministic sequence that can be replaced with object-relative logic
    once perception + pose estimates are wired in.
    """

    def __init__(self, dt: float, steps_per_phase: int = 120) -> None:
        self._dt = dt
        self._steps_per_phase = steps_per_phase
        self._phases = self._default_phases()
        self._phase_index = 0
        self._phase_step = 0

    def _default_phases(self) -> list[ExpertPhase]:
        pick = np.array([0.35, 0.10, 0.05], dtype=np.float32)
        place = np.array([0.50, -0.10, 0.08], dtype=np.float32)
        return self._build_phases(pick, place)

    def _build_phases(self, pick: np.ndarray, place: np.ndarray) -> list[ExpertPhase]:
        pregrasp = pick + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        grasp = pick
        lift = pick + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        retreat = place + np.array([0.0, 0.0, 0.12], dtype=np.float32)

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return [
            ExpertPhase("pregrasp", pregrasp, quat, False, self._steps_per_phase),
            ExpertPhase("grasp", grasp, quat, False, self._steps_per_phase),
            ExpertPhase("close", grasp, quat, True, self._steps_per_phase // 2),
            ExpertPhase("lift", lift, quat, True, self._steps_per_phase),
            ExpertPhase("place", place, quat, True, self._steps_per_phase),
            ExpertPhase("open", place, quat, False, self._steps_per_phase // 2),
            ExpertPhase("retreat", retreat, quat, False, self._steps_per_phase),
        ]

    def reset(self) -> None:
        self._phase_index = 0
        self._phase_step = 0

    def set_pick_place(self, pick: np.ndarray, place: np.ndarray) -> None:
        self._phases = self._build_phases(pick, place)
        self.reset()

    def step(self) -> ExpertPhase:
        phase = self._phases[self._phase_index]
        self._phase_step += 1
        if self._phase_step >= phase.duration_steps:
            self._phase_index = (self._phase_index + 1) % len(self._phases)
            self._phase_step = 0
        return phase
