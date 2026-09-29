"""Strict-LIFO runtime lifecycle: guaranteed teardown on exit, fault and signal.

Ascend bring-up holds device resources that the Python process must release in
a specific order -- outstanding stream work must be synchronized before the
stream object is dropped, workspaces and arenas must be released before the
allocator cache is flushed, and ``torch.npu.empty_cache()`` / device reset must
happen only after every owner is gone. :class:`RuntimeLifecycleManager` makes
that order *structural* instead of procedural:

* every resource is registered with a name, a kind and a release callback;
  :meth:`teardown` pops the stack in strict reverse-registration (LIFO) order
  and always runs to completion -- a failing step is recorded and skipped,
  never propagated, so later (earlier-registered) steps still run;
* teardown is idempotent and reentrancy-safe: a signal arriving mid-teardown
  is recorded, not re-executed;
* the manager wires itself into ``atexit`` and traps ``SIGINT``/``SIGTERM`` so
  unexpected exits take the same path as normal ones (signal handlers chain to
  any previously installed handler, then default to ``KeyboardInterrupt`` /
  ``SystemExit(128 + signum)``);
* once the resource stack is drained, a device epilogue runs
  ``synchronize_device`` -> ``release_cache`` -> ``reset_device`` through the
  :class:`~dsv4_moe_runtime.hardware.runtime.DeviceRuntime` seam -- on NPU
  that is ``torch.npu.synchronize`` / ``empty_cache`` / accumulated-stat
  reset; on CPU it is a no-op.
"""

from __future__ import annotations

import atexit
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from dsv4_moe_runtime.hardware.runtime import DeviceRuntime

TRAPPED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
TeardownCallback = Callable[[], None]


@dataclass(frozen=True)
class TeardownStep:
    """One executed (or failed) release action."""

    name: str
    kind: str  # stream | workspace | arena | device | custom
    ok: bool
    error: str | None = None
    duration_us: int = 0


@dataclass
class TeardownReport:
    """Audit record of one teardown pass."""

    trigger: str
    steps: list[TeardownStep] = field(default_factory=list)
    registration_order: list[str] = field(default_factory=list)

    @property
    def lifo_order_respected(self) -> bool:
        """Non-device steps ran in exact reverse-registration order."""
        resource_steps = [step.name for step in self.steps if not step.kind.startswith("device")]
        return resource_steps == list(reversed(self.registration_order))

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps) and self.lifo_order_respected

    @property
    def errors(self) -> list[str]:
        return [f"{step.name}: {step.error}" for step in self.steps if step.error is not None]


@dataclass
class _Registration:
    name: str
    kind: str
    teardown: TeardownCallback
    refs: list[object] = field(default_factory=list)
    released: bool = False

    def release(self) -> TeardownStep:
        start = time.perf_counter()
        try:
            self.teardown()
            return TeardownStep(
                name=self.name,
                kind=self.kind,
                ok=True,
                duration_us=int((time.perf_counter() - start) * 1e6),
            )
        except Exception as exc:  # noqa: BLE001  -- teardown must never abort the chain
            return TeardownStep(
                name=self.name,
                kind=self.kind,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_us=int((time.perf_counter() - start) * 1e6),
            )
        finally:
            self.released = True
            self.refs.clear()  # drop manager-held resource references


class RuntimeLifecycleManager:
    """Owns the teardown order of every device resource in one harness run."""

    def __init__(self, runtime: DeviceRuntime | None = None, *, label: str = "dsv4", trap_signals: bool = True):
        self._runtime = runtime
        self._label = label
        self._stack: list[_Registration] = []
        self._registration_order: list[str] = []
        self._report: TeardownReport | None = None
        self._tearing_down = False
        self._previous_handlers: dict[int, object] = {}
        self._atexit_hook = self._atexit_teardown
        atexit.register(self._atexit_hook)
        if trap_signals:
            self._install_signal_traps()

    # ---------------------------------------------------------- registration

    def push(
        self, name: str, teardown: TeardownCallback, *, kind: str = "custom", refs: list[object] | None = None
    ) -> None:
        """Register one release action; it runs last-registered-first."""
        if self._report is not None:
            raise RuntimeError(f"lifecycle {self._label!r} already torn down; cannot register {name!r}")
        if any(registration.name == name for registration in self._stack):
            raise ValueError(f"lifecycle {self._label!r} already holds a resource named {name!r}")
        self._stack.append(_Registration(name=name, kind=kind, teardown=teardown, refs=list(refs or [])))
        self._registration_order.append(name)

    def register_stream(self, name: str, stream: object | None) -> None:
        """Synchronize the stream, then drop it (CANN reclaims unreferenced streams)."""

        def teardown() -> None:
            if stream is not None and self._runtime is not None:
                self._runtime.synchronize_stream(stream)

        self.push(name, teardown, kind="stream", refs=[stream])

    def register_workspace(
        self, name: str, release: TeardownCallback | None = None, *, workspace: object | None = None
    ) -> None:
        """Release a kernel workspace (aclnn two-call protocol buffers)."""
        self.push(name, release or (lambda: None), kind="workspace", refs=[workspace])

    def register_arena(
        self, name: str, release: TeardownCallback | None = None, *, arena: object | None = None
    ) -> None:
        """Release an AOT arena (slot pool / scratchpad backing tensors)."""
        self.push(name, release or (lambda: None), kind="arena", refs=[arena])

    # -------------------------------------------------------------- sessions

    @contextmanager
    def session(self) -> Iterator[RuntimeLifecycleManager]:
        """Teardown on scope exit -- normal (``exit``) or exceptional (``exception:*``)."""
        try:
            yield self
        except BaseException as exc:
            self.teardown(trigger=f"exception:{type(exc).__name__}")
            raise
        else:
            self.teardown(trigger="exit")

    # -------------------------------------------------------------- teardown

    @property
    def last_report(self) -> TeardownReport | None:
        return self._report

    @property
    def active_resources(self) -> list[str]:
        return [registration.name for registration in self._stack]

    def teardown(self, *, trigger: str = "manual") -> TeardownReport:
        """Drain the resource stack in strict LIFO order, then reset the device.

        Idempotent (subsequent calls return the first report) and
        reentrancy-safe (a signal during teardown is absorbed).
        """
        if self._report is not None:
            return self._report
        if self._tearing_down:  # signal arrived mid-teardown: absorb it
            return TeardownReport(trigger=f"{trigger}:suppressed-during-teardown")
        self._tearing_down = True
        self.uninstall_signal_traps()  # restore default signal semantics first

        steps: list[TeardownStep] = []
        while self._stack:
            steps.append(self._stack.pop().release())  # strict LIFO by construction
        if self._runtime is not None:
            steps.append(_run_device_step("device/synchronize", self._runtime.synchronize_device))
            steps.append(_run_device_step("device/release-cache", self._runtime.release_cache))
            reset = getattr(self._runtime, "reset_device", None)
            if callable(reset):
                steps.append(_run_device_step("device/reset", reset))

        self._report = TeardownReport(trigger=trigger, steps=steps, registration_order=list(self._registration_order))
        atexit.unregister(self._atexit_hook)
        return self._report

    # --------------------------------------------------------------- signals

    def _install_signal_traps(self) -> None:
        for signum in TRAPPED_SIGNALS:
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._make_trap(signum, previous))
                self._previous_handlers[signum] = previous
            except (ValueError, OSError, RuntimeError):
                # Not the main thread, or the platform refuses the install:
                # atexit still guarantees teardown.
                continue

    def uninstall_signal_traps(self) -> None:
        """Restore the handlers that were active before the traps were installed."""
        for signum, previous in self._previous_handlers.items():
            try:
                signal.signal(signum, previous)  # type: ignore[arg-type]
            except (ValueError, OSError, RuntimeError):
                continue
        self._previous_handlers.clear()

    def _make_trap(self, signum: int, previous: object) -> Callable[[int, object], None]:
        def trap(frame_signum: int, frame: object) -> None:
            self.teardown(trigger=f"signal:{frame_signum}")
            if callable(previous):
                previous(frame_signum, frame)  # chain user/pytest handlers
            elif frame_signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(128 + frame_signum)

        return trap

    def _atexit_teardown(self) -> None:
        self.teardown(trigger="atexit")


def _run_device_step(name: str, action: Callable[[], None]) -> TeardownStep:
    return _Registration(name=name, kind="device", teardown=action).release()
