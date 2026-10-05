"""Concurrency and transition gating rules."""

from __future__ import annotations

from enum import Enum


class Action(Enum):
    ADMIT = "admit"  # hand the head request to the running engine
    SWAP = "swap"  # engine is idle and on the wrong model: restart it
    WAIT = "wait"


class SchedulerPolicy:
    """Strict FIFO with a drain barrier.

    The oldest waiting request decides what happens next. Requests for the
    running model are admitted (up to `max_concurrency` at once). A request for
    another model blocks everything behind it until the engine has drained,
    then triggers a swap. FIFO order prevents starvation of either model.
    """

    def __init__(self, max_concurrency: int = 1) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency

    def decide(self, head_model: str, active_model: str | None, in_flight: int) -> Action:
        if active_model == head_model:
            return Action.ADMIT if in_flight < self.max_concurrency else Action.WAIT
        return Action.SWAP if in_flight == 0 else Action.WAIT
