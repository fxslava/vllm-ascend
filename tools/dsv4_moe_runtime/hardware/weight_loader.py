"""DirectStorage-style streaming weight ingestion (disk -> pinned DDR -> HBM).

Bypasses the OS page cache where the platform allows it and streams file
bytes through a *fixed* pool of pinned host chunks into device (HBM) views:

* **Unbuffered backend** -- Linux ``O_DIRECT`` + ``os.preadv`` directly into
  chunk memory; Windows ``ReadFile`` with ``FILE_FLAG_NO_BUFFERING`` via
  ctypes, positional through an ``OVERLAPPED`` block, straight into the
  chunk's ``data_ptr``. Both require 4096-byte-aligned file offsets, transfer
  lengths and chunk base addresses.
* **Mmap backend** -- portable ``mmap`` windows aligned to the allocation
  granularity; the copy still lands in pinned chunks.

Allocation contract: the chunk pool is allocated once in ``__init__``
(optionally supplied from outside -- e.g. ``AscendPinnedHostStorage`` or
``ExchangeBuffer`` pages -- provided every chunk base is 4096-aligned for the
unbuffered backend). ``stream_into`` performs only positional reads into
chunk memory and in-place ``copy_`` into the destination view; no tensor is
created after construction. A mis-probed backend degrades to mmap, never to
dynamic allocation.
"""

from __future__ import annotations

import ctypes
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
        self._tail_fd: int | None = None
        self._tail_handle: int | None = None
        self._kernel32_tail = None
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
            os.preadv(self._fd, [view], aligned_offset)

    def mmap_view(self, file_offset: int, num_bytes: int) -> torch.Tensor:
        """Zero-copy read-only tensor view over the file bytes (kept alive)."""
        aligned_offset = align_down(file_offset, mmap.ALLOCATIONGRANULARITY)
        mapped_len = min(
            align_up(file_offset - aligned_offset + num_bytes),
            max(self._size - aligned_offset, 0),
        )
        if mapped_len < num_bytes:
            raise OSError(f"mmap window short: need {num_bytes} bytes at {file_offset}")
        window = mmap.mmap(self._fd, length=mapped_len, access=mmap.ACCESS_READ, offset=aligned_offset)
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
            os.preadv(self._tail_fd, [view], file_offset)

    def close(self) -> None:
        if self._win_handle is not None:
            self._kernel32.CloseHandle(ctypes.c_void_p(self._win_handle))
            self._win_handle = None
        if self._tail_handle is not None:
            self._kernel32_tail.CloseHandle(ctypes.c_void_p(self._tail_handle))
            self._tail_handle = None
        if self._tail_fd is not None:
            os.close(self._tail_fd)
            self._tail_fd = None
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
