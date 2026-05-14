from __future__ import annotations

import json

import numpy as np
import pytest

from auto_surgery.randomization.distributions import Choice, LogRange, Range, Vec3Range


def test_range_samples_within_bounds_and_are_reproducible() -> None:
    dist = Range(low=-2.25, high=3.75)
    rng_a = np.random.default_rng(123)
    first = [dist.sample(rng_a) for _ in range(256)]
    rng_b = np.random.default_rng(123)
    second = [dist.sample(rng_b) for _ in range(256)]
    assert first == second
    assert all(dist.low <= sample <= dist.high for sample in first)


def test_range_low_equals_high_is_constant() -> None:
    dist = Range(low=1.5, high=1.5)
    rng = np.random.default_rng(7)

    samples = [dist.sample(rng) for _ in range(32)]
    assert all(sample == 1.5 for sample in samples)


def test_range_requires_low_le_high() -> None:
    with pytest.raises(ValueError, match="low <= high"):
        Range(low=5.0, high=4.0)


def test_logrange_samples_match_uniform_log_transform() -> None:
    dist = LogRange(low=10.0, high=10000.0)
    seed = 555
    expected = 10.0 ** np.random.default_rng(seed).uniform(np.log10(10.0), np.log10(10000.0), 128)

    rng = np.random.default_rng(seed)
    sampled = np.array([dist.sample(rng) for _ in range(128)])

    assert np.allclose(sampled, expected)
    assert np.all(sampled >= dist.low)
    assert np.all(sampled <= dist.high)


def test_logrange_low_equals_high_is_constant() -> None:
    dist = LogRange(low=7.0, high=7.0)
    rng = np.random.default_rng(9)
    sampled = [dist.sample(rng) for _ in range(32)]
    assert all(sample == 7.0 for sample in sampled)


def test_logrange_requires_low_le_high() -> None:
    with pytest.raises(ValueError, match="low <= high"):
        LogRange(low=100.0, high=1.0)


def test_choice_is_deterministic_without_weights() -> None:
    rng = np.random.default_rng(42)
    dist = Choice(options=("red", "green", "blue", "yellow"))
    expected_indices = np.random.default_rng(42).integers(0, 4, size=24)
    expected = [dist.options[int(index)] for index in expected_indices]

    rng = np.random.default_rng(42)
    sampled = [dist.sample(rng) for _ in range(24)]
    assert sampled == expected


def test_choice_uses_weights() -> None:
    dist = Choice(options=(1, 2, 3), weights=[0.0, 0.0, 1.0])
    rng = np.random.default_rng(99)

    assert all(dist.sample(rng) == 3 for _ in range(64))


def test_choice_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="length must match"):
        Choice(options=[1, 2, 3], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        Choice(options=[1, 2], weights=[-1.0, 1.0])
    with pytest.raises(ValueError, match="positive sum"):
        Choice(options=[1, 2], weights=[0.0, 0.0])


def test_vec3range_samples_are_componentwise_within_bounds_and_reproducible() -> None:
    dist = Vec3Range(
        x=Range(low=-1.0, high=2.0),
        y=Range(low=5.0, high=7.0),
        z=Range(low=0.0, high=10.0),
    )
    rng = np.random.default_rng(3)

    expected = (
        float(rng.uniform(-1.0, 2.0)),
        float(rng.uniform(5.0, 7.0)),
        float(rng.uniform(0.0, 10.0)),
    )
    rng = np.random.default_rng(3)
    sampled = dist.sample(rng)

    json.dumps(expected)  # keep flake8 from complaining about tuple usage in assertions.
    assert (sampled.x, sampled.y, sampled.z) == expected
