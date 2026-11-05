"""Local, persistent page owners and restart-safe execution metadata.

Only the coordinator writes run-level metadata. Page functions are imported in
their owner process, and a phase returns only after every submitted page has
finished or has an explicit failure. A phase is therefore a safe rendezvous for
the few decisions that really depend on all pages.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_write_json(path: Path | str, data: Any) -> None:
    """Commit a complete JSON document without exposing a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default).encode()).hexdigest()


def fingerprint_paths(paths: list[Path] | tuple[Path, ...], extra: Any = None) -> str:
    """Hash explicitly selected inputs; never crawl a run or inspect config files."""
    records = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"name": str(path.resolve()), "sha256": _file_digest(path)})
    return _fingerprint({"files": records, "extra": extra})


fingerprint_files = fingerprint_paths


def _output_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path.resolve()), "kind": "file", "sha256": _file_digest(path)}
    if path.is_dir():
        return {"path": str(path.resolve()), "kind": "directory"}
    raise FileNotFoundError(f"checkpoint required output is missing: {path}")


def run_checkpoint(
    output_dir: Path | str,
    step: str,
    fingerprint: Any,
    required_outputs: list[Path] | tuple[Path, ...] | Callable[[dict[str, Any]], list[Path]],
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Reuse a completed step only while its inputs and required outputs match.

    The callback owns its page directory. External image-task/critic journals
    remain authoritative during interrupted steps; a checkpoint does not retry
    an incomplete remote submission on its own.
    """
    if not step or Path(step).name != step or step in {".", ".."}:
        raise ValueError("checkpoint step must be one filename component")
    path = Path(output_dir) / "checkpoints" / f"{step}.json"
    expected = _fingerprint(fingerprint)
    def output_records(result: dict[str, Any]) -> list[dict[str, Any]]:
        outputs = required_outputs(result) if callable(required_outputs) else required_outputs
        return [_output_record(Path(item)) for item in outputs]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "completed" and record.get("fingerprint") == expected:
            current = output_records(record["result"])
            if current == record.get("outputs"):
                _checkpoint_event(step, reused=True)
                return record["result"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    started = time.time()
    atomic_write_json(path, {"status": "running", "fingerprint": expected, "started_at": started})
    try:
        result = fn()
        records = output_records(result)
        atomic_write_json(path, {"status": "completed", "fingerprint": expected,
                                "started_at": started, "finished_at": time.time(),
                                "outputs": records, "result": result})
        _checkpoint_event(step, reused=False)
        return result
    except BaseException as exc:
        # Error messages can include request bodies; persist only the type here.
        atomic_write_json(path, {"status": "failed", "fingerprint": expected,
                                "started_at": started, "finished_at": time.time(),
                                "error_type": type(exc).__name__})
        raise


class RunLock:
    """Exclusive ownership of one output directory across cooperating invocations."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir).resolve()
        self._stream: Any = None

    def __enter__(self) -> "RunLock":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self.output_dir / ".page_execution.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            self._stream = None
            raise RuntimeError(f"output directory is already in use: {self.output_dir}") from exc
        self._stream.seek(0)
        self._stream.truncate()
        json.dump({"pid": os.getpid(), "started_at": time.time()}, self._stream)
        self._stream.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


_RESOURCE_SEMAPHORES: dict[str, Any] = {}
_OWNER_ID: int | None = None
_OWNER_OUTBOX: Any = None
_ACTIVE_JOB: dict[str, Any] = {}
_LEASE_DEPTH = threading.local()


def _prepare_child_process(parent_pid: int) -> None:
    """Give each worker its own process group and stop it if its parent dies."""
    os.setsid()

    def terminate_group(_signum: int, _frame: Any) -> None:
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, terminate_group)
    if os.name == "posix" and Path("/proc/self").exists():
        import ctypes
        # Linux PR_SET_PDEATHSIG. No CUDA or application state is inherited.
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)
        if os.getppid() != parent_pid:
            terminate_group(signal.SIGTERM, None)


def _terminate_owned_process(process: Any) -> None:
    """Kill only this worker's session, including its browser/office children."""
    if process.pid is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _checkpoint_event(step: str, *, reused: bool) -> None:
    if _OWNER_OUTBOX is not None:
        _bounded_put(_OWNER_OUTBOX, {"kind": "checkpoint", **_ACTIVE_JOB,
                                    "step": step, "reused": reused, "finished_at": time.time()})


@contextlib.contextmanager
def resource_lease(name: str) -> Iterator[None]:
    """Bound expensive local helpers; each operation releases its own permit."""
    semaphore = _RESOURCE_SEMAPHORES.get(name)
    depths = getattr(_LEASE_DEPTH, "resources", None)
    if depths is None:
        depths = _LEASE_DEPTH.resources = {}
    if semaphore is None or depths.get(name, 0):
        yield
    else:
        semaphore.acquire()
        depths[name] = 1
        try:
            yield
        finally:
            depths.pop(name, None)
            semaphore.release()


def current_owner_id() -> int | None:
    return _OWNER_ID


def _bounded_put(destination: Any, message: dict[str, Any]) -> None:
    """Keep each pipe write atomic even if its worker is killed mid-send."""
    if len(pickle.dumps(message, protocol=4)) + 4 > 4096:
        raise ValueError("worker notification exceeds the bounded IPC message size")
    destination.put(message)


def _owner_main(owner_id: int, inbox: Any, outbox: Any, gpu_request: Any,
                gpu_reply: Any, stop: Any, gpu_stop: Any, resources: dict[str, Any],
                output_dir: str, ipc_root: str, parent_pid: int) -> None:
    global _OWNER_ID, _RESOURCE_SEMAPHORES, _OWNER_OUTBOX, _ACTIVE_JOB
    _OWNER_ID = owner_id
    _prepare_child_process(parent_pid)
    _RESOURCE_SEMAPHORES = resources
    _OWNER_OUTBOX = outbox
    from .gpu_runtime import install_gpu_broker
    install_gpu_broker(owner_id, gpu_request, gpu_reply, stop, gpu_stop, ipc_root)
    _bounded_put(outbox, {"kind": "ready", "owner_id": owner_id, "pid": os.getpid()})
    while not stop.is_set():
        try:
            command = inbox.get(timeout=0.25)
        except queue.Empty:
            continue
        if command is None:
            return
        job = command["job"]
        started = time.time()
        event = {"phase": command["phase"], "task_id": command["task_id"],
                 "slide_id": job["slide_id"], "original_index": job["original_index"],
                 "owner_id": owner_id, "pid": os.getpid()}
        _ACTIVE_JOB = event
        _bounded_put(outbox, {"kind": "started", **event, "started_at": started})
        try:
            module_name, function_name = command["entrypoint"].split(":", 1)
            function = getattr(importlib.import_module(module_name), function_name)
            claim_dir = Path(output_dir) / ".page_claims" / hashlib.sha256(job["slide_id"].encode()).hexdigest()
            with RunLock(claim_dir):
                result = function(job)
            relative = f"owner_{owner_id}/{command['task_id'].replace(':', '_')}.result.json"
            atomic_write_json(Path(ipc_root) / relative, result)
            _bounded_put(outbox, {"kind": "completed", **event, "started_at": started,
                                 "finished_at": time.time(), "result_file": relative})
        except BaseException as exc:
            relative = f"owner_{owner_id}/{command['task_id'].replace(':', '_')}.error.json"
            atomic_write_json(Path(ipc_root) / relative, {"error_type": type(exc).__name__, "error": str(exc)})
            _bounded_put(outbox, {"kind": "failed", **event, "started_at": started,
                                 "finished_at": time.time(), "error_type": type(exc).__name__,
                                 "error_file": relative})


class PagePhaseError(RuntimeError):
    def __init__(self, phase: str, results: list[Any], errors: list[dict[str, Any]]):
        self.phase, self.results, self.errors = phase, results, errors
        super().__init__(f"page phase {phase!r} failed for {len(errors)} page(s): "
                         + ", ".join(str(item.get("slide_id", "?")) for item in errors))


class PageExecutor:
    """Five persistent owners, with stable ownership across true deck barriers."""

    def __init__(self, output_dir: Path | str, workers: int = 5,
                 gpu_devices: list[str] | tuple[str, ...] | str | None = None,
                 config: Any = None, mock_diffusion: bool = False,
                 *, run_lock: bool = True, startup_timeout: float = 120.0):
        if workers < 1:
            raise ValueError("page workers must be at least 1")
        self.output_dir = Path(output_dir).resolve()
        self.workers = workers
        self.config, self.mock_diffusion = config, mock_diffusion
        self.gpu_devices = gpu_devices
        self.startup_timeout = startup_timeout
        self._take_lock = run_lock
        self._lock: RunLock | None = None
        self._ctx = mp.get_context("spawn")
        self._owners: dict[int, Any] = {}
        self._inboxes: dict[int, Any] = {}
        self._outboxes: dict[int, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._active = False
        self._phase_counter = 0
        self._ready_owners: set[int] = set()
        self._gpu_pool: Any = None

    def __enter__(self) -> "PageExecutor":
        if self._active:
            raise RuntimeError("executor is already active")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._take_lock:
            self._lock = RunLock(self.output_dir)
            self._lock.__enter__()
        try:
            self._stop = self._ctx.Event()
            self._gpu_stop = self._ctx.Event()
            self._ipc_root = self.output_dir / ".page_ipc" / uuid.uuid4().hex
            self._gpu_requests = [self._ctx.Queue() for _ in range(self.workers)]
            self._gpu_replies = [self._ctx.Queue() for _ in range(self.workers)]
            self._resources = {name: self._ctx.BoundedSemaphore(self.workers) for name in ("browser", "render")}
            if self.config is not None:
                from .gpu_runtime import GpuPool
                self._gpu_pool = GpuPool(ctx=self._ctx, config=self.config,
                                         devices=self.gpu_devices, max_workers=self.workers,
                                         mock=self.mock_diffusion, request_queues=self._gpu_requests,
                                         replies=self._gpu_replies, stop=self._gpu_stop,
                                         ipc_root=self._ipc_root)
                self._gpu_pool.start()
            else:
                # A non-diffusion phase needs no CUDA model or GPU subprocess.
                self._gpu_stop.set()
            self._active = True
            return self
        except BaseException:
            self.close()
            raise

    def _start_owner(self, owner_id: int) -> None:
        self._ready_owners.discard(owner_id)
        if owner_id in self._owners:
            self._owners[owner_id].join(timeout=0.1)
            self._inboxes[owner_id].cancel_join_thread()
            self._inboxes[owner_id].close()
            self._outboxes[owner_id].cancel_join_thread()
            self._outboxes[owner_id].close()
        inbox = self._ctx.Queue()
        outbox = self._ctx.Queue()
        process = self._ctx.Process(target=_owner_main,
            args=(owner_id, inbox, outbox, self._gpu_requests[owner_id], self._gpu_replies[owner_id],
                  self._stop, self._gpu_stop, self._resources, str(self.output_dir), str(self._ipc_root), os.getpid()),
            name=f"rl4ppt-page-{owner_id}")
        process.start()
        self._inboxes[owner_id], self._owners[owner_id] = inbox, process
        self._outboxes[owner_id] = outbox

    def _next_event(self) -> dict[str, Any]:
        for outbox in self._outboxes.values():
            try:
                return outbox.get_nowait()
            except queue.Empty:
                continue
        time.sleep(0.02)
        raise queue.Empty

    def _record(self, event: dict[str, Any]) -> None:
        # Business results/errors may contain content; timing records contain only
        # bounded execution metadata and are written by this single coordinator.
        timing = {key: value for key, value in event.items()
                  if key in {"kind", "phase", "task_id", "slide_id", "original_index", "owner_id",
                             "pid", "started_at", "finished_at", "error_type", "step", "reused"}}
        self._events.append(timing)
        atomic_write_json(self.output_dir / "page_execution.json", {
            "schema_version": 1, "page_workers": self.workers,
            "actual_page_owners": len(self._owners),
            "gpu_workers": self._gpu_pool.worker_count if self._gpu_pool else 0,
            "checkpoint_reused": sum(item.get("kind") == "checkpoint" and item.get("reused") is True for item in self._events),
            "checkpoint_executed": sum(item.get("kind") == "checkpoint" and item.get("reused") is False for item in self._events),
            "events": self._events})

    def run_phase(self, entrypoint: str, jobs: list[dict[str, Any]], phase: str = "page") -> list[Any]:
        if not self._active:
            raise RuntimeError("use PageExecutor as a context manager")
        if ":" not in entrypoint:
            raise ValueError("entrypoint must be module:function")
        ids = [job.get("slide_id") for job in jobs]
        indices = [job.get("original_index") for job in jobs]
        if len(set(ids)) != len(ids) or len(set(indices)) != len(indices):
            raise ValueError("duplicate page ownership in one phase")
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError("each page needs a nonnegative original_index")
        if any(not isinstance(sid, str) or not sid for sid in ids):
            raise ValueError("each page needs a slide_id")
        self._phase_counter += 1
        commands = []
        for index, job in enumerate(jobs):
            owner = job["original_index"] % self.workers
            if owner not in self._owners or not self._owners[owner].is_alive():
                self._start_owner(owner)
            commands.append({"task_id": f"{self._phase_counter}:{index}", "phase": phase,
                             "entrypoint": entrypoint, "job": job, "owner_id": owner})
        # Queue all assigned pages. No owner waits at a deck barrier while it
        # still has unstarted pages; the coordinator calls the next phase later.
        pending = {item["task_id"]: item for item in commands}
        positions = {item["task_id"]: index for index, item in enumerate(commands)}
        for command in commands:
            self._inboxes[command["owner_id"]].put(command)
        results: list[Any] = [None] * len(jobs)
        errors: list[dict[str, Any]] = []
        waiting_started = time.monotonic()
        while pending:
            if self._gpu_pool is not None:
                self._gpu_pool.poll_health()
            try:
                event = self._next_event()
                if event["kind"] == "ready":
                    self._ready_owners.add(event["owner_id"])
                elif event.get("task_id") in pending:
                    self._record(event)
                    if event["kind"] in {"completed", "failed"}:
                        task_id = event["task_id"]
                        if event["kind"] == "completed":
                            results[positions[task_id]] = json.loads((self._ipc_root / event["result_file"]).read_text(encoding="utf-8"))
                        else:
                            event.update(json.loads((self._ipc_root / event["error_file"]).read_text(encoding="utf-8")))
                            errors.append(event)
                        del pending[task_id]
            except queue.Empty:
                pass
            for owner_id in {command["owner_id"] for command in pending.values()}:
                process = self._owners[owner_id]
                if not process.is_alive() or (owner_id not in self._ready_owners and time.monotonic() - waiting_started > self.startup_timeout):
                    _terminate_owned_process(process)
                    for task_id, command in list(pending.items()):
                        if command["owner_id"] == owner_id:
                            error = {"kind": "failed", "phase": phase, "task_id": task_id,
                                     "owner_id": owner_id, "slide_id": command["job"]["slide_id"],
                                     "original_index": command["job"]["original_index"],
                                     "error_type": "PageWorkerExited", "exitcode": process.exitcode,
                                     "finished_at": time.time()}
                            errors.append(error)
                            self._record(error)
                            del pending[task_id]
        if errors:
            raise PagePhaseError(phase, results, errors)
        return results

    def close(self) -> None:
        if hasattr(self, "_stop"):
            self._stop.set()
        for owner, inbox in self._inboxes.items():
            if self._owners[owner].is_alive():
                inbox.put(None)
        for process in self._owners.values():
            process.join(timeout=2)
            _terminate_owned_process(process)
        if self._gpu_pool is not None:
            self._gpu_pool.close()
        queues = [*self._inboxes.values(), *self._outboxes.values(), *getattr(self, "_gpu_replies", []),
                  *getattr(self, "_gpu_requests", [])]
        for item in queues:
            if item is not None:
                # Cancel feeder joins after children are drained; abandoned jobs
                # from a terminated owner must never hang interpreter shutdown.
                item.cancel_join_thread()
                item.close()
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None
        self._active = False

    def __exit__(self, *_: Any) -> None:
        self.close()
