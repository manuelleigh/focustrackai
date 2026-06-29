from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Generic, Iterable, TypeVar

T = TypeVar("T")


class TemporalConsensus(Generic[T]):
    def __init__(self, window_size: int, min_votes: int):
        self.window_size = max(1, int(window_size))
        self.min_votes = max(1, int(min_votes))
        self.values: Deque[T] = deque(maxlen=self.window_size)
        self.stable_value: T | None = None

    def update(self, candidate: T) -> T:
        self.values.append(candidate)
        counts = Counter(self.values)
        value, votes = counts.most_common(1)[0]
        if votes >= self.min_votes:
            self.stable_value = value
        elif self.stable_value is None:
            self.stable_value = candidate
        return self.stable_value

    def snapshot(self) -> list[T]:
        return list(self.values)

    @classmethod
    def from_iterable(cls, values: Iterable[T], window_size: int, min_votes: int) -> "TemporalConsensus[T]":
        consensus = cls(window_size=window_size, min_votes=min_votes)
        for value in values:
            consensus.update(value)
        return consensus
