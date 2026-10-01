"""DirectStorage-style streaming weight ingestion (disk -> pinned DDR -> HBM).

Bypasses the OS page cache where the platform allows it and streams file
bytes through a *fixed* pool of pinned host chunks into device (HBM) views:

* **Unbuffered backend** -- Linux ``O_DIRECT`` + ``os.preadv`` directly into
  chunk memory; Windows ``ReadFile`` with ``FILE_FLAG_NO_BUFFERING`` via
  ctypes, positional through an ``OVERLAPPED`` block, straight into the
  chunk's ``data_ptr``. Both require 4096-byte-aligned file offsets, transfer
  lengths and chunk base addresses.
* **Mmap backend** -- portable ``mmap`` windows aligned to the allocation
  granularity; the copy still lands in pinned chunks. Its positional reads go
  through ``_read_at``, which uses ``os.preadv`` where it exists and
  ``io.FileIO.readinto`` where it does not -- Windows has no ``preadv`` at all,
  and this is the backend the degrade-on-probe-failure path runs on.

Upstream safetensors exporters align tensor spans to 8 bytes, so in practice
*no* span begins on a 4096-byte sector: every unbuffered read is widened down
to the enclosing sector and the valid bytes are sliced out of the chunk. A read
that comes back short is an error rather than a partial fill, because the bytes
it did not write are whatever the recycled chunk last held.

Allocation contract: the chunk pool is allocated once in ``__init__``
(optionally supplied from outside -- e.g. ``AscendPinnedHostStorage`` or
``ExchangeBuffer`` pages -- provided every chunk base is 4096-aligned for the
unbuffered backend). ``stream_into`` performs only positional reads into
chunk memory and in-place ``copy_`` into the destination view; no tensor is
created after construction. A backend that fails its probe degrades to mmap,
never to dynamic allocation.
"""

from __future__ import annotations

import ctypes
import io
import mmap
import os
import platform
from collections.abc import Sequence

import torch

from .runtime import DeviceRuntime

IO_ALIGNMENT = 4096
DEFAULT_DENSE_CHUNK_BYTES = 256 * 1024 * 1024
DEFAULT_DENSE_CHUNKS = 8  # 8 x 256 MiB = 2 GiB fixed staging pool
DEFAULT_EXPERT_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_EXPERT_CHUNKS = 4

MODE_AUTO = "auto"
MODE_UNBUFFERED = "unbuffered"
MODE_MMAP = "mmap"

#: ``os.preadv`` is POSIX-only; Windows has no positional-read syscall binding.
_HAS_PREADV = hasattr(os, "preadv")

_WIN32_GENERIC_READ = 0x80000000
_WIN32_SHARE_READ = 1
_WIN32_OPEN_EXISTING = 3
_WIN32_FLAG_NO_BUFFERING = 0x20000000
_WIN32_ATTRIBUTE_NORMAL = 0x80


class StreamingBackpressureError(RuntimeError):
    """Raised when streaming would require growing the fixed chunk pool."""


def align_up(value: int, alignment: int = IO_ALIGNMENT) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def align_down(value: int, alignment: int = IO_ALIGNMENT) -> int:
    return value - (value % alignment)


def _is_page_aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % IO_ALIGNMENT == 0


class _Overlapped(ctypes.Structure):
    """Windows positional-read block (synchronous handle + OVERLAPPED offset)."""

    _fields_ = [
        ("internal", ctypes.c_void_p),
        ("internal_high", ctypes.c_void_p),
        ("offset", ctypes.c_uint32),
        ("offset_high", ctypes.c_uint32),
        ("event", ctypes.c_void_p),
    ]


class _RawFileReader:
    """Positional byte-source abstraction over the three IO backends."""

    def __init__(self, path: str, mode: str):
        self._path = path
        self._mode = mode
        self._fd: int | None = None
        self._win_handle: int | None = None
        self._kernel32 = None
        self._mmap_fd: int | None = None
        self._tail_fd: int | None = None
        self._tail_handle: int | None = None
        self._kernel32_tail = None
        self._seek_streams: dict[int, io.FileIO] = {}
        self._size = os.path.getsize(path)
        self._open()

    @property
    def backend(self) -> str:
        return self._mode

    @property
    def size(self) -> int:
        return self._size

    def _open(self) -> None:
        if self._mode == MODE_UNBUFFERED and platform.system() == "Windows":
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = self._kernel32.CreateFileW(
                self._path,
                _WIN32_GENERIC_READ,
                _WIN32_SHARE_READ,
                None,
                _WIN32_OPEN_EXISTING,
                _WIN32_FLAG_NO_BUFFERING,
                None,
            )
            if handle is None or handle == -1:
                raise OSError(f"CreateFileW failed for {self._path}")
            self._win_handle = handle
        elif self._mode == MODE_UNBUFFERED:
            self._fd = os.open(self._path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        else:
            self._fd = os.open(self._path, os.O_RDONLY)

    def _read_at(self, fd: int, view: memoryview, offset: int) -> int:
        """Fill ``view`` from ``offset``, allocating nothing, and report the count.

        ``os.preadv`` is POSIX-only -- it does not exist on Windows at all -- so
        the descriptor path cannot rest on it. That matters beyond tidiness: the
        Windows ``FILE_FLAG_NO_BUFFERING`` handle is a separate branch, and this
        one is what the documented *degrade to mmap, never to dynamic
        allocation* fallback runs on. Without a portable implementation the
        fallback raised ``AttributeError`` on Windows the moment a filesystem
        refused unbuffered IO, which is the one situation it exists to cover.

        ``readinto`` over an ``io.FileIO`` wrapper is the portable equivalent and
        writes straight into the chunk. The loop is there because both backends
        may return short; the caller still checks the total against what the
        file can supply, because a short read leaves the rest of the chunk
        holding the *previous* expert's bytes.
        """
        if _HAS_PREADV:
            return os.preadv(fd, [view], offset)
        stream = self._seek_streams.get(fd)
        if stream is None:
            # closefd=False: the descriptor's owner is close(), not this wrapper.
            stream = io.FileIO(fd, mode="rb", closefd=False)
            self._seek_streams[fd] = stream
        stream.seek(offset)
        total = 0
        while total < len(view):
            got = stream.readinto(view[total:])
            if not got:
                break
            total += got
        return total

    def read_into(self, chunk: torch.Tensor, aligned_offset: int, aligned_length: int) -> None:
        """Aligned positional read filling ``chunk[:aligned_length]``.

        The chunk must hold ``aligned_length`` bytes; the caller slices the
        valid head out of the chunk after the read (alignment head stays).
        """
        if aligned_length > chunk.numel():
            raise StreamingBackpressureError(f"chunk capacity {chunk.numel()} < aligned transfer {aligned_length}")
        if self._win_handle is not None:
            overlapped = _Overlapped(offset=aligned_offset & 0xFFFFFFFF, offset_high=aligned_offset >> 32)
            read = ctypes.c_uint32(0)
            ok = self._kernel32.ReadFile(
                ctypes.c_void_p(self._win_handle),
                ctypes.c_void_p(chunk.data_ptr()),
                ctypes.c_uint32(aligned_length),
                ctypes.byref(read),
                ctypes.byref(overlapped),
            )
            expected = min(aligned_length, max(self._size - aligned_offset, 0))
            if not ok or read.value < expected:
                raise OSError(f"ReadFile short read at {aligned_offset}: {read.value}/{expected}")
        elif self._fd is not None:
            view = memoryview(chunk.numpy())[:aligned_length]
            read = self._read_at(self._fd, view, aligned_offset)
            expected = min(aligned_length, max(self._size - aligned_offset, 0))
            if read < expected:
                raise OSError(f"positional read short at {aligned_offset}: {read}/{expected}")

    def _mapping_fd(self) -> int:
        """Descriptor usable for ``mmap``.

        The Windows unbuffered backend owns a raw ``HANDLE`` and no descriptor
        at all, so mapping needs its own buffered fd (opened once, lazily);
        mapping ``None`` would silently map the pagefile instead of the file.
        """
        if self._fd is not None:
            return self._fd
        if self._mmap_fd is None:
            self._mmap_fd = os.open(self._path, os.O_RDONLY)
        return self._mmap_fd

    def mmap_view(self, file_offset: int, num_bytes: int) -> torch.Tensor:
        """Zero-copy read-only tensor view over the file bytes (kept alive)."""
        aligned_offset = align_down(file_offset, mmap.ALLOCATIONGRANULARITY)
        mapped_len = min(
            align_up(file_offset - aligned_offset + num_bytes),
            max(self._size - aligned_offset, 0),
        )
        if mapped_len < num_bytes:
            raise OSError(f"mmap window short: need {num_bytes} bytes at {file_offset}")
        window = mmap.mmap(self._mapping_fd(), length=mapped_len, access=mmap.ACCESS_READ, offset=aligned_offset)
        skip = file_offset - aligned_offset
        view = torch.frombuffer(window, dtype=torch.uint8)[skip : skip + num_bytes]
        view._owning_window = window  # type: ignore[attr-defined]  # pins the mapping
        return view

    def read_tail_into(self, chunk: torch.Tensor, file_offset: int, num_bytes: int) -> None:
        """Buffered positional read for the sub-sector EOF tail.

        Unbuffered IO cannot deliver the final partial sector of a file; the
        tail falls back to a small page-cached read into chunk memory (a few
        KiB once per stream end, never a pool resize).
        """
        if self._tail_fd is None and self._tail_handle is None:
            if platform.system() == "Windows":
                self._kernel32_tail = ctypes.WinDLL("kernel32", use_last_error=True)
                handle = self._kernel32_tail.CreateFileW(
                    self._path,
                    _WIN32_GENERIC_READ,
                    _WIN32_SHARE_READ,
                    None,
                    _WIN32_OPEN_EXISTING,
                    _WIN32_ATTRIBUTE_NORMAL,
                    None,
                )
                if handle is None or handle == -1:
                    raise OSError(f"tail CreateFileW failed for {self._path}")
                self._tail_handle = handle
            else:
                self._tail_fd = os.open(self._path, os.O_RDONLY)
        if self._tail_handle is not None:
            overlapped = _Overlapped(offset=file_offset & 0xFFFFFFFF, offset_high=file_offset >> 32)
            read = ctypes.c_uint32(0)
            ok = self._kernel32_tail.ReadFile(
                ctypes.c_void_p(self._tail_handle),
                ctypes.c_void_p(chunk.data_ptr()),
                ctypes.c_uint32(num_bytes),
                ctypes.byref(read),
                ctypes.byref(overlapped),
            )
            if not ok or read.value != num_bytes:
                raise OSError(f"tail ReadFile short at {file_offset}: {read.value}/{num_bytes}")
        else:
            view = memoryview(chunk.numpy())[:num_bytes]
            read = self._read_at(self._tail_fd, view, file_offset)
            if read != num_bytes:
                raise OSError(f"tail positional read short at {file_offset}: {read}/{num_bytes}")

    def close(self) -> None:
        for stream in self._seek_streams.values():
            stream.close()  # closefd=False, so the descriptor itself survives to below
        self._seek_streams.clear()
        if self._win_handle is not None:
            self._kernel32.CloseHandle(ctypes.c_void_p(self._win_handle))
            self._win_handle = None
        if self._tail_handle is not None:
            self._kernel32_tail.CloseHandle(ctypes.c_void_p(self._tail_handle))
            self._tail_handle = None
        if self._tail_fd is not None:
            os.close(self._tail_fd)
            self._tail_fd = None
        if self._mmap_fd is not None:
            os.close(self._mmap_fd)
            self._mmap_fd = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class StreamingWeightLoader:
    """Fixed-pool pinned streaming reader with a pipelined copy leg.

    ``chunk_pool`` may be supplied externally (``AscendPinnedHostStorage`` /
    ``ExchangeBuffer`` pages); otherwise ``num_chunks x chunk_bytes`` of host
    memory is allocated once (pinned when the runtime supports it). Every
    chunk must be 4096-aligned for the unbuffered backends.
    """

    def __init__(
        self,
        runtime: DeviceRuntime,
        file_path: str,
        chunk_bytes: int = DEFAULT_DENSE_CHUNK_BYTES,
        num_chunks: int = DEFAULT_DENSE_CHUNKS,
        io_mode: str = MODE_AUTO,
        chunk_pool: Sequence[torch.Tensor] | None = None,
        pinned: bool | None = None,
    ):
        if chunk_bytes % IO_ALIGNMENT:
            raise ValueError(f"chunk_bytes {chunk_bytes} must be 4096-aligned")
        self._runtime = runtime
        self._stream = runtime.make_stream()
        self._chunks = self._build_chunks(chunk_pool, chunk_bytes, num_chunks, runtime, pinned)
        if any(not _is_page_aligned(chunk) for chunk in self._chunks) and io_mode == MODE_UNBUFFERED:
            raise ValueError("unbuffered IO requires 4096-aligned chunk buffers")
        self._reader, self._backend = self._open_reader(file_path, io_mode)
        self._chunk_bytes = chunk_bytes
        self._in_flight = 0
        self.bytes_read = 0
        self.window_count = 0

    @staticmethod
    def _build_chunks(
        chunk_pool: Sequence[torch.Tensor] | None,
        chunk_bytes: int,
        num_chunks: int,
        runtime: DeviceRuntime,
        pinned: bool | None,
    ) -> list[torch.Tensor]:
        if chunk_pool is not None:
            for chunk in chunk_pool:
                if chunk.numel() < chunk_bytes:
                    raise ValueError("external chunk pool cannot hold chunk_bytes")
            return list(chunk_pool)
        want_pin = runtime.supports_pinned_host_memory if pinned is None else pinned
        if want_pin and not runtime.supports_pinned_host_memory:
            raise ValueError("pinned chunks requested but unsupported by this runtime")
        flags: dict[str, object] = {"pin_memory": True} if want_pin else {}
        return [torch.empty(chunk_bytes, dtype=torch.uint8, **flags) for _ in range(num_chunks)]

    def _open_reader(self, path: str, io_mode: str) -> tuple[_RawFileReader, str]:
        if io_mode == MODE_AUTO:
            for candidate in (MODE_UNBUFFERED, MODE_MMAP):
                try:
                    reader = _RawFileReader(path, candidate)
                    if candidate == MODE_UNBUFFERED:
                        reader.read_into(self._chunks[0], 0, IO_ALIGNMENT)
                        reader.close()
                        reader = _RawFileReader(path, candidate)  # fresh positional cursor
                    return reader, candidate
                except OSError:
                    continue
            raise OSError(f"no usable IO backend for {path}")
        return _RawFileReader(path, io_mode), io_mode

    @property
    def backend(self) -> str:
        return self._reader.backend

    @property
    def file_size(self) -> int:
        return self._reader.size

    @property
    def chunk_bytes(self) -> int:
        return self._chunk_bytes

    def mmap_view(self, file_offset: int, num_bytes: int) -> torch.Tensor:
        """Zero-copy host view over a file span (used by the blocking provider path)."""
        return self._reader.mmap_view(file_offset, num_bytes)

    def stream_into(self, destination: torch.Tensor, file_offset: int, num_bytes: int) -> int:
        """Stream ``num_bytes`` at ``file_offset`` into the flat destination view.

        Pipelined: up to ``len(chunks)`` read+copy windows are outstanding
        before the ring forces a stream synchronization. Returns bytes moved.
        """
        if destination.numel() < num_bytes:
            raise ValueError(f"destination holds {destination.numel()} bytes < {num_bytes}")
        flat = destination.view(-1)[:num_bytes]
        moved = 0
        offset = file_offset
        # Every copy is issued on this loader's own stream, which is also the
        # stream the ring wrap and the final sync wait on. Issuing them on the
        # caller's current stream instead would leave a chunk free to be
        # overwritten by the next host read while its H2D DMA is still running.
        with self._runtime.stream_context(self._stream):
            while moved < num_bytes:
                remaining = num_bytes - moved
                aligned_offset = align_down(offset)
                skip = offset - aligned_offset
                # Unbuffered reads deliver whole 4096-byte sectors in file space;
                # boundary sectors are re-read by the next window, the sub-sector
                # EOF remainder falls back to the buffered tail path.
                chunk = self._chunks[self._in_flight % len(self._chunks)]
                self._in_flight += 1
                if self._in_flight > len(self._chunks):
                    self._runtime.synchronize_stream(self._stream)  # ring wrap: recycle
                valid = min(remaining, self._chunk_bytes - skip)
                read_len = align_down(skip + valid, IO_ALIGNMENT)
                if read_len > 0:
                    self._reader.read_into(chunk, aligned_offset, read_len)
                    covered = read_len - skip  # valid bytes inside the full sectors
                    flat[moved : moved + covered].copy_(chunk[skip : skip + covered], non_blocking=True)
                    moved += covered
                    offset += covered
                    self.bytes_read += covered
                    self.window_count += 1
                else:  # sub-sector remainder: buffered tail path (page-cached)
                    tail_len = min(remaining, max(self._reader.size - offset, 0))
                    self._reader.read_tail_into(chunk, offset, tail_len)
                    flat[moved : moved + tail_len].copy_(chunk[:tail_len], non_blocking=True)
                    moved += tail_len
                    offset += tail_len
                    self.bytes_read += tail_len
        self._runtime.synchronize_stream(self._stream)
        return moved

    def synchronize(self) -> None:
        self._runtime.synchronize_stream(self._stream)

    def close(self) -> None:
        self._reader.close()
