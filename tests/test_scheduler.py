import asyncio

from coliollama.core.engine.process import EngineStartError, ModelTarget
from coliollama.core.scheduler.policy import Action, SchedulerPolicy
from coliollama.core.scheduler.queue_manager import QueueManager


class FakeLifecycle:
    def __init__(self, fail=()):
        self.active_model = None
        self.events = []
        self.fail = set(fail)

    base_url = "http://fake"

    async def start(self, target):
        if self.active_model:
            self.events.append(f"stop {self.active_model}")
            self.active_model = None
        await asyncio.sleep(0.01)
        if target.name in self.fail:
            raise EngineStartError("nope")
        self.events.append(f"start {target.name}")
        self.active_model = target.name

    async def stop(self):
        self.active_model = None
        return True


A, B = ModelTarget("a", "/a"), ModelTarget("b", "/b")


def test_policy():
    p = SchedulerPolicy(2)
    assert p.decide("a", "a", 1) is Action.ADMIT
    assert p.decide("a", "a", 2) is Action.WAIT
    assert p.decide("b", "a", 1) is Action.WAIT
    assert p.decide("b", "a", 0) is Action.SWAP
    assert p.decide("a", None, 0) is Action.SWAP


def test_same_model_sequential_and_swap_after_drain():
    async def main():
        lc = FakeLifecycle()
        qm = QueueManager(lc, SchedulerPolicy(1))
        order = []

        async def job(target, tag, hold=0.05):
            lease = await qm.acquire(target)
            order.append(f"begin {tag}")
            await asyncio.sleep(hold)
            order.append(f"end {tag}")
            lease.release()

        t1 = asyncio.create_task(job(A, "a1", 0.3))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(job(A, "a2"))
        t3 = asyncio.create_task(job(B, "b1"))
        t4 = asyncio.create_task(job(A, "a3"))  # arrives after B: must queue behind it
        await asyncio.sleep(0.01)
        assert qm.snapshot()["waiting"] == 3
        await asyncio.gather(t1, t2, t3, t4)
        assert order == ["begin a1", "end a1", "begin a2", "end a2",
                         "begin b1", "end b1", "begin a3", "end a3"]
        assert lc.events == ["start a", "stop a", "start b", "stop b", "start a"]

    asyncio.run(main())


def test_concurrency_limit_two():
    async def main():
        qm = QueueManager(FakeLifecycle(), SchedulerPolicy(2))
        leases = [await qm.acquire(A) for _ in range(2)]
        third = asyncio.create_task(qm.acquire(A))
        await asyncio.sleep(0.05)
        assert not third.done() and qm.snapshot()["in_flight"] == 2
        leases[0].release()
        (await third).release()
        leases[1].release()
        assert qm.snapshot()["queue_depth"] == 0

    asyncio.run(main())


def test_start_failure_fails_only_that_model_and_cancel_cleans_up():
    async def main():
        qm = QueueManager(FakeLifecycle(fail={"b"}), SchedulerPolicy(1))
        lease = await qm.acquire(A)
        bad = asyncio.create_task(qm.acquire(B))
        cancelled = asyncio.create_task(qm.acquire(A))
        await asyncio.sleep(0.02)
        cancelled.cancel()
        lease.release()
        try:
            await bad
            raise AssertionError("expected failure")
        except EngineStartError:
            pass
        assert qm.snapshot()["waiting"] == 0
        (await qm.acquire(A)).release()

    asyncio.run(main())
