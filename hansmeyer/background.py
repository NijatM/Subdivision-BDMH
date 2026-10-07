"""Heavy work for the app in a worker process (full-depth bakes, print preparation), so the window never freezes.

Why a process: Polyscope's render loop holds Python's GIL, so a thread barely runs while the window is open.
Why shared memory: the pipe between processes is read by a helper thread that also needs the GIL for every
chunk, so under the render loop a 100 MB mesh arrives at ~10 MB/s. Here the large arrays of a job's
arguments and of its result travel through one shared-memory block each, and the pipe carries one small
message.

The executor's helper threads (they forward a job and pick up its result) still need the GIL a few times per
job, and the app's frame callback keeps retaking it, which starves them for seconds: while a Worker is
busy(), the app pauses for a millisecond each frame (see app.ui).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import pickle
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import shared_memory

INLINE = 1 << 20  # payloads with less array data than this simply go through the pipe
_created: list = []  # blocks this process made: closed by release() once the other side has read them


def share(obj):
    """A small, picklable message carrying `obj`; its numpy arrays go into one shared-memory block."""
    buffers = []
    data = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    raws = [b.raw() for b in buffers]
    total = sum(r.nbytes for r in raws)
    if total < INLINE:
        return ("pickle", pickle.dumps(obj, protocol=5))
    block = shared_memory.SharedMemory(create=True, size=total)
    spans, offset = [], 0
    for r in raws:
        block.buf[offset:offset + r.nbytes] = r
        spans.append((offset, r.nbytes))
        offset += r.nbytes
    del raws, buffers
    _created.append(block)
    return ("shm", data, block.name, spans)


def unshare(message):
    """The object a share() message carries (its arrays copied out, the block freed)."""
    if message[0] == "pickle":
        return pickle.loads(message[1])
    _, data, name, spans = message
    block = shared_memory.SharedMemory(name=name)
    try:
        buffers = [bytearray(block.buf[o:o + n]) for o, n in spans]
    finally:
        block.close()
        try:
            block.unlink()
        except FileNotFoundError:
            pass
    return pickle.loads(data, buffers=buffers)


def release():
    """Close the blocks this process shared (the reader unlinks them; Windows frees a block with its last handle)."""
    while _created:
        try:
            _created.pop().close()
        except OSError:
            pass


def _job(fn, message):
    release()  # the previous result has been read by now
    return share(fn(*unshare(message)))


class Worker:
    """One background process. submit() returns a Future; once it is done(), result() gives fn's return value
    (or raises its exception, or BrokenProcessPool if the process died, e.g. out of memory)."""

    def __init__(self):
        self._executor = None
        self._pending: set = set()

    def start(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            atexit.register(self.stop)
        return self

    def submit(self, fn, *args) -> Future:
        future = self.start()._executor.submit(_job, fn, share(args))
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)
        return future

    def busy(self) -> bool:
        """A job is queued or running (or its result is not picked up yet by the helper thread)."""
        return bool(self._pending)

    def warm_up(self):
        """Start the process now (imports take a moment), so the first real job doesn't wait for it."""
        self.submit(_ready)

    def result(self, future: Future):
        try:
            return unshare(future.result())
        finally:
            release()

    def broken(self):
        """Forget a dead process: the next submit() starts a fresh one."""
        self._executor = None

    def stop(self):
        """End the process now, even mid-job (the app is closing)."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        procs = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()


def _ready():
    return True


# ----------------------------------------------------------------- baking
_pipe = None
_plugins = None


def bake(design: dict, depth: int, face_budget: int, root: str):
    """Run a design's schedule to `depth` (in the worker; it keeps its own level cache between bakes)."""
    global _pipe, _plugins
    from . import functions
    from .pipeline import Pipeline
    from .schedule import Design

    folder = os.path.join(root, "functions")
    plugins = sorted((f, os.path.getmtime(os.path.join(folder, f))) for f in os.listdir(folder)
                     if f.endswith(".py")) if os.path.isdir(folder) else []
    if _pipe is None or _pipe.root != root or plugins != _plugins:
        functions.load_plugins(folder)  # the app's functions/ plug-ins, as loaded there
        _pipe, _plugins = Pipeline(root=root, max_bytes=600_000_000), plugins
    _pipe.face_budget = face_budget
    return _pipe.run(Design.from_dict(design), depth)
