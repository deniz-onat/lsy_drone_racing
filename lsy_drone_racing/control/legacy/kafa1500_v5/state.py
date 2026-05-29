"""KaFa_1500_v5 denetleyicisi için gözlem ayrıştırma işlemleri."""
# ruff: noqa: TC002

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DroneObservation:
    """[qx, qy, qz, qw] kuaterniyon sırasını kullanan tek dron gözlem çerçevesi."""

    target_gate: int
    gate_pos: NDArray[np.float64]
    gate_quat: NDArray[np.float64]
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    quat: NDArray[np.float64]


def scalar_gate_index(value: object) -> int:
    """Depodaki target-gate alanını tekil bir tamsayıya dönüştürür."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"target_gate must contain exactly one value, got shape {arr.shape}")
    return int(arr.reshape(()))


def as_vector(value: object, length: int, name: str) -> NDArray[np.float64]:
    """Sabit uzunlukta bir kayan nokta vektörü döndürür."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


def parse_observation(obs: dict[str, NDArray[np.floating]]) -> DroneObservation:
    """Denetleyicinin ihtiyaç duyduğu gözlem anahtarlarını ayrıştırır ve doğrular."""
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
    gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
    if gate_pos.ndim != 2 or gate_pos.shape[1] != 3:
        raise ValueError(f"gates_pos must have shape (n, 3), got {gate_pos.shape}")
    if gate_quat.ndim != 2 or gate_quat.shape[1] != 4:
        raise ValueError(f"gates_quat must have shape (n, 4), got {gate_quat.shape}")
    if len(gate_pos) < 4 or len(gate_quat) < 4:
        raise ValueError("KaFa_1500_v5 expects at least four gates")

    return DroneObservation(
        target_gate=scalar_gate_index(obs["target_gate"]),
        gate_pos=gate_pos[:4].copy(),
        gate_quat=gate_quat[:4].copy(),
        pos=as_vector(obs["pos"], 3, "pos"),
        vel=as_vector(obs["vel"], 3, "vel"),
        quat=as_vector(obs["quat"], 4, "quat"),
    )
