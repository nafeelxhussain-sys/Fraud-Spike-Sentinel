from dataclasses import dataclass
from typing import List


@dataclass
class SpikeResult:
    is_spike: bool
    current_rate: float
    baseline_mean: float
    baseline_std: float
    z_score: float


def detect_spike(
    recent_scores: List[float],
    window: int = 10,
    baseline_window: int = 50,
    z_threshold: float = 3.0
    ):

    """
    recent_scores: risk scores (0-1) for an accounts transactions in
    chronological order (oldest first).
    window: how many of the most recent scores count as current activity.
    baseline_window: how many scores before that count as normal for
    this account.
    """
    if len(recent_scores) < window + 5:
        # not enough history to say anything meaningful yet
        return SpikeResult(False, current_rate=_safe_mean(recent_scores), baseline_mean=0.0, baseline_std=0.0, z_score=0.0)

    current = recent_scores[-window:]
    baseline_pool = recent_scores[:-window][-baseline_window:]

    if len(baseline_pool) < 5:
        return SpikeResult(False, current_rate=_safe_mean(current), baseline_mean=0.0, baseline_std=0.0, z_score=0.0)

    current_rate = _safe_mean(current)
    baseline_mean = _safe_mean(baseline_pool)
    baseline_std = _safe_std(baseline_pool, baseline_mean)

    # floor the std so a perfectly flat history doesn't produce a divide
    # by near-zero and flag every tiny fluctuation as a "spike"
    denom = max(baseline_std, 0.02)
    z = (current_rate - baseline_mean) / denom

    return SpikeResult(
        is_spike= z >= z_threshold,
        current_rate=current_rate,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        z_score=z,
    )


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_std(values, mean):
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    # bezzels correction
    return variance ** 0.5
