"""Request queues for the active model and the pending transition."""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass

from coliollama.core.engine.lifecycle import EngineLifecycle
from coliollama.core.engine.process import ModelTarget
from coliollama.core.scheduler.policy import Action, SchedulerPolicy


class Lease:
    """Permission to talk to the engine; must be released exactly when done."""

    def __init__(self, manager: QueueManager, ticket: _Ticket) -> None:
        self._manager = manager
        self._ticket = ticket

    @property
    def base_url(self) -> str:
        return self._manager.lifecycle.base_url

    def release(self) -> None:
        self._manager._release(self._ticket)

    async def __aenter__(self) -> Lease:
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()


@dataclass(eq=False)
class _Ticket:
    target: ModelTarget
    future: asyncio.Future
    admitted: bool = False
    released: bool = False


class QueueManager:
    def __init__(self, lifecycle: EngineLifecycle, policy: SchedulerPolicy) -> None:
        self.lifecycle = lifecycle
        self.policy = policy
        self._waiting: deque[_Ticket] = deque()
        self._in_flight = 0
        self._transition: asyncio.Task | None = None
        self._transition_target: str | None = None
        self._closed = False

    async def acquire(self, target: ModelTarget) -> Lease:
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        ticket = _Ticket(target, asyncio.get_running_loop().create_future())
        self._waiting.append(ticket)
        self._pump()
        try:
            return await ticket.future
        except BaseException:
            if ticket in self._waiting:
                self._waiting.remove(ticket)
            elif ticket.admitted:
                self._release(ticket)
            self._pump()
            raise

    def _release(self, ticket: _Ticket) -> None:
        if ticket.released or not ticket.admitted:
            return
        ticket.released = True
        self._in_flight -= 1
        self._pump()

    def _pump(self) -> None:
        while self._waiting and self._transition is None:
            head = self._waiting[0]
            if head.future.done():  # cancelled while queued
                self._waiting.popleft()
                continue
            action = self.policy.decide(
                head.target.name, self.lifecycle.active_model, self._in_flight
            )
            if action is Action.ADMIT:
                self._waiting.popleft()
                head.admitted = True
                self._in_flight += 1
                head.future.set_result(Lease(self, head))
            elif action is Action.SWAP:
                self._transition_target = head.target.name
                self._transition = asyncio.ensure_future(self._swap(head.target))
                return
            else:
                return

    async def _swap(self, target: ModelTarget) -> None:
        try:
            await self.lifecycle.start(target)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Fail only the requests that were waiting for this model.
            failed = [t for t in self._waiting if t.target.name == target.name]
            for ticket in failed:
                self._waiting.remove(ticket)
                if not ticket.future.done():
                    ticket.future.set_exception(exc)
        finally:
            self._transition = None
            self._transition_target = None
            self._pump()

    def snapshot(self) -> dict:
        per_model = Counter(t.target.name for t in self._waiting if not t.future.done())
        waiting = sum(per_model.values())
        return {
            "active_model": self.lifecycle.active_model,
            "loading_model": self._transition_target,
            "in_flight": self._in_flight,
            "waiting": waiting,
            "queue_depth": self._in_flight + waiting,
            "waiting_by_model": dict(per_model),
        }

    async def stop_engine(self) -> bool:
        """Force-terminate the engine now. In-flight requests will fail."""
        return await self.lifecycle.stop()

    async def shutdown(self) -> None:
        self._closed = True
        for ticket in list(self._waiting):
            if not ticket.future.done():
                ticket.future.set_exception(RuntimeError("scheduler is shutting down"))
        self._waiting.clear()
        if self._transition is not None:
            self._transition.cancel()
            await asyncio.gather(self._transition, return_exceptions=True)
        await self.lifecycle.stop()
