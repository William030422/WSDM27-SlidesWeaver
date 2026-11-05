"""A local request queue with one reusable Diffusion process per selected GPU.

Page owners never import torch. They send strings, paths and scalar settings to
these processes; model state and CUDA tensors never cross a process boundary.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any


_BROKER: dict[str, Any] | None = None


def install_gpu_broker(owner_id: int, request: Any, reply: Any, stop: Any, gpu_stop: Any, ipc_root: str) -> None:
    global _BROKER
    _BROKER = {"owner_id": owner_id, "request": request, "reply": reply,
               "stop": stop, "gpu_stop": gpu_stop, "ipc_root": Path(ipc_root)}


def _diffusion_settings(config: Any) -> dict[str, Any]:
    return {"model_path": str(config.diffusion_model_path),
            "width": int(config.diffusion_size.width), "height": int(config.diffusion_size.height),
            "steps": int(config.diffusion_steps), "guidance_scale": float(config.diffusion_guidance_scale)}


def generate_diffusion_image(*, config: Any, prompt: str, output_path: Path | str, seed: int) -> dict[str, Any]:
    """Generate one image through this page owner's pool, or fail explicitly."""
    broker = _BROKER
    if broker is None:
        raise RuntimeError("Diffusion requests require an active PageExecutor owner")
    if broker["gpu_stop"].is_set():
        raise RuntimeError("local GPU pool is unavailable")
    request_id = uuid.uuid4().hex
    submitted = time.time()
    deadline = time.monotonic() + float(getattr(config, "gpu_timeout_seconds", 1800.0))
    from .page_runtime import atomic_write_json, _bounded_put
    relative = f"owner_{broker['owner_id']}/gpu/{request_id}.json"
    atomic_write_json(broker["ipc_root"] / relative, {
        "request_id": request_id, "owner_id": broker["owner_id"],
        "prompt": prompt, "output_path": str(Path(output_path).resolve()),
        "seed": int(seed), "settings": _diffusion_settings(config), "submitted_at": submitted})
    _bounded_put(broker["request"], {"request_id": request_id, "request_file": relative})
    while not broker["stop"].is_set():
        try:
            result = broker["reply"].get(timeout=0.25)
        except queue.Empty:
            if broker["gpu_stop"].is_set():
                raise RuntimeError("local GPU worker exited while this page was waiting")
            if time.monotonic() > deadline:
                raise TimeoutError("local Diffusion request exceeded its execution deadline")
            continue
        if result.get("request_id") != request_id:
            # A previously interrupted request may finish after its owner was
            # replaced. Its output is retained, but it cannot satisfy this job.
            continue
        if result.get("status") != "succeeded":
            raise RuntimeError(f"local Diffusion failed ({result.get('error_type', 'unknown')})")
        return result["result"]
    raise RuntimeError("page execution stopped while waiting for Diffusion")


def _selected_devices(devices: Any, max_workers: int, mock: bool) -> list[str]:
    if isinstance(devices, str):
        devices = [value.strip() for value in devices.split(",") if value.strip()]
    if devices is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mock:
            devices = ["mock:0"]
        elif inherited is not None:
            devices = [value.strip() for value in inherited.split(",") if value.strip() and value.strip() != "-1"]
        else:
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=True, timeout=10)
                devices = result.stdout.splitlines()
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError("no GPU devices discovered; specify devices or use mock diffusion") from exc
    devices = [str(item).strip() for item in devices]
    if not devices or any(not item for item in devices):
        raise ValueError("at least one GPU device is required")
    if len(devices) != len(set(devices)):
        raise ValueError("GPU device list contains duplicates")
    return devices[:max_workers]


def _gpu_lock(device: str) -> Any:
    """Use physical UUIDs so index/UUID aliases cannot bypass cooperating locks."""
    identity = device
    try:
        result = subprocess.run(["nvidia-smi", f"--id={device}", "--query-gpu=uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=10)
        identities = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(identities) != 1:
            raise RuntimeError("GPU selector must identify one physical GPU")
        identity = identities[0]
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot resolve GPU device {device}") from exc
    directory = Path(tempfile.gettempdir()) / f"rl4ppt-gpu-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stream = (directory / f"{hashlib.sha256(identity.encode()).hexdigest()}.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"GPU {device} is already reserved by another RL4PPT run") from exc
    return stream


def _save_mock(path: Path, width: int, height: int, seed: int) -> None:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 12, height // 6, width * 5 // 12, height // 3), fill=(225, 225, 225))
    draw.rectangle((width * 7 // 12, height // 5, width * 11 // 12, height * 4 // 5),
                   fill=(180 + seed % 60, 200, 210))
    image.save(path, format="PNG")


def _gpu_main(device: str, settings: dict[str, Any], mock: bool, requests: Any,
              replies: list[Any], stop: Any, readiness: Any, parent_pid: int) -> None:
    from .page_runtime import _prepare_child_process
    _prepare_child_process(parent_pid)
    lock = None
    pipe = None
    torch = None
    try:
        if not mock:
            lock = _gpu_lock(device)
            # This happens before importing torch/diffusers in this process.
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        readiness.put({"kind": "ready", "device": device, "pid": os.getpid()})
        while not stop.is_set():
            try:
                request = requests.get(timeout=0.25)
            except queue.Empty:
                continue
            if request is None:
                return
            started = time.time()
            readiness.put({"kind": "request_started", "pid": os.getpid(),
                           "request_id": request["request_id"], "owner_id": request["owner_id"]})
            message = {"request_id": request["request_id"], "status": "failed"}
            temporary: str | None = None
            try:
                if request["settings"] != settings:
                    raise ValueError("Diffusion settings do not match this run's frozen GPU model")
                output = Path(request["output_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".png", dir=output.parent)
                os.close(descriptor)
                load_started = time.perf_counter()
                cold_load_seconds = 0.0
                if not mock and pipe is None:
                    import torch as torch_module
                    from diffusers import ZImagePipeline
                    torch = torch_module
                    if not torch.cuda.is_available():
                        raise RuntimeError("Diffusion generation requires CUDA")
                    pipe = ZImagePipeline.from_pretrained(settings["model_path"], torch_dtype=torch.bfloat16,
                                                         low_cpu_mem_usage=False, local_files_only=True)
                    pipe.to("cuda:0")
                    cold_load_seconds = time.perf_counter() - load_started
                inference_started = time.perf_counter()
                if mock:
                    _save_mock(Path(temporary), settings["width"], settings["height"], request["seed"])
                    peak_memory = 0.0
                else:
                    torch.cuda.reset_peak_memory_stats(0)
                    image = pipe(prompt=request["prompt"], height=settings["height"], width=settings["width"],
                                 num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                                 generator=torch.Generator(device="cuda:0").manual_seed(request["seed"])).images[0]
                    torch.cuda.synchronize(0)
                    image.convert("RGB").save(temporary, format="PNG")
                    peak_memory = round(torch.cuda.max_memory_allocated(0) / 1024**3, 3)
                os.replace(temporary, output)
                temporary = None
                message.update(status="succeeded", result={
                    "output": str(output.resolve()), "seed": request["seed"], "device": device,
                    "gpu_pid": os.getpid(), "mock": mock,
                    "elapsed_seconds": round(time.perf_counter() - inference_started, 6),
                    "cold_load_seconds": round(cold_load_seconds, 6),
                    "queue_seconds": round(max(0.0, started - request["submitted_at"]), 6),
                    "peak_gpu_memory_gib": peak_memory})
            except BaseException as exc:
                message["error_type"] = type(exc).__name__
                # A failed page does not turn unrelated queued pages into mock
                # images. Release the allocator cache after a recoverable OOM.
                if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            finally:
                if temporary is not None:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temporary)
            replies[request["owner_id"]].put(message)
            readiness.put({"kind": "request_finished", "pid": os.getpid(),
                           "request_id": request["request_id"]})
    except BaseException as exc:
        readiness.put({"kind": "failed", "device": device, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if lock is not None:
            lock.close()


class GpuPool:
    def __init__(self, *, ctx: Any, config: Any, devices: Any, max_workers: int,
                 mock: bool, request_queues: list[Any], replies: list[Any], stop: Any, ipc_root: Path):
        self.ctx, self.settings, self.mock = ctx, _diffusion_settings(config), mock
        self.devices = _selected_devices(devices, max_workers, mock)
        self.requests, self.replies, self.stop = request_queues, replies, stop
        self.ipc_root = ipc_root
        self.processes: list[Any] = []
        self.inboxes: dict[int, Any] = {}
        self.event_queues: dict[int, Any] = {}
        self.pending: Any = deque()
        self.active_requests: dict[int, dict[str, Any]] = {}
        self.reported_dead: set[int] = set()
        self.worker_count = len(self.devices)

    def start(self) -> None:
        try:
            for device in self.devices:
                inbox, events = self.ctx.Queue(), self.ctx.Queue()
                process = self.ctx.Process(target=_gpu_main,
                    args=(device, self.settings, self.mock, inbox, self.replies, self.stop, events, os.getpid()),
                    name=f"rl4ppt-gpu-{device}")
                process.start()
                self.processes.append(process)
                self.inboxes[process.pid], self.event_queues[process.pid] = inbox, events
            ready = set()
            deadline = time.monotonic() + 60
            while len(ready) < self.worker_count:
                for events in self.event_queues.values():
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        continue
                    if event["kind"] == "failed":
                        raise RuntimeError(f"GPU worker startup failed: {event.get('error', event['error_type'])}")
                    if event["kind"] == "ready":
                        ready.add(event["device"])
                if self.has_dead_worker() or time.monotonic() > deadline:
                    raise RuntimeError("GPU worker exited or timed out during startup")
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def has_dead_worker(self) -> bool:
        return any(not process.is_alive() for process in self.processes)

    def poll_health(self) -> None:
        """Dispatch to independent GPU inboxes and contain a card's hard exit.

        Multiple killable consumers must not share multiprocessing.Queue's read
        lock: a killed idle consumer can otherwise block every healthy card.
        The coordinator is the sole consumer of the owner request queue.
        """
        for events in self.event_queues.values():
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                if event["kind"] == "request_finished":
                    current = self.active_requests.get(event["pid"], {})
                    if current.get("request_id") == event["request_id"]:
                        self.active_requests.pop(event["pid"], None)
        for process in self.processes:
            if process.is_alive() or process.pid in self.reported_dead:
                continue
            self.reported_dead.add(process.pid)
            pending = self.active_requests.pop(process.pid, None)
            if pending is not None:
                self.replies[pending["owner_id"]].put({"status": "failed",
                    "request_id": pending["request_id"], "error_type": "GpuWorkerExited"})
        if not any(process.is_alive() for process in self.processes):
            self.stop.set()
            return
        for requests in self.requests:
            while True:
                try:
                    notice = requests.get_nowait()
                except queue.Empty:
                    break
                # Owners publish complete request files before a tiny, atomic
                # notice. A killed long-prompt sender cannot poison other pages.
                request = json.loads((self.ipc_root / notice["request_file"]).read_text(encoding="utf-8"))
                self.pending.append(request)
        for process in self.processes:
            if self.pending and process.is_alive() and process.pid not in self.active_requests:
                request = self.pending.popleft()
                self.active_requests[process.pid] = request
                self.inboxes[process.pid].put(request)

    def close(self) -> None:
        from .page_runtime import _terminate_owned_process
        self.stop.set()
        for process in self.processes:
            if process.is_alive():
                self.inboxes[process.pid].put(None)
        for process in self.processes:
            process.join(timeout=2)
            _terminate_owned_process(process)
        for item in [*self.inboxes.values(), *self.event_queues.values()]:
            item.cancel_join_thread()
            item.close()
