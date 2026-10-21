"""Parsing helpers for reproducible multi-seed experiments."""

from __future__ import annotations


DEFAULT_SEED_SPEC = "0-9"


def parse_seed_spec(spec: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges such as ``0-9``."""

    values: list[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Empty seed token in {spec!r}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"Invalid seed range {token!r}")
            start, stop = (int(part) for part in parts)
            if start > stop:
                raise ValueError(f"Seed range must be ascending: {token!r}")
            values.extend(range(start, stop + 1))
        elif token.isdigit():
            values.append(int(token))
        else:
            raise ValueError(f"Invalid seed {token!r}")
    if not values:
        raise ValueError("At least one seed is required")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate seeds are not allowed: {spec!r}")
    return tuple(values)


def canonical_seed_spec(seeds: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(seed) for seed in seeds)
