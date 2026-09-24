r"""Hold system RAM until only a target amount is left available, and keep it.

Why: a measurement taken on a starved box looks exactly like a slow device.
On 2026-09-23 the laptop had 0.8 GB free with the pagefile working, and three
figures measured that evening were later withdrawn or failed to reproduce --
decode speed, a probe verdict, and a 9-25x side-lane slowdown [OBSERVED
2026-09-23/24, d944b78, OPENCODE-PLAN.md]. This makes that condition
reproducible on purpose, so "was it the RAM?" gets tested instead of argued.

    venv\Scripts\python scripts\ram-hog.py --leave-gb 0.8

It allocates in chunks, writes to every page so the memory is really
resident, re-touches everything every few seconds so the OS has to page
something else out, and prints "READY" once available RAM is at the target.
Runs until killed. Windows only (GlobalMemoryStatusEx).

Out: status lines on stdout. Refuses to go past --max-gb, and gives a chunk
back if available RAM drops under 0.3 GB, so a run cannot wedge the box.
"""
import argparse
import ctypes
import sys
import time

CHUNK = 256 * 1024 * 1024
PAGE = 4096


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_gb():
    """Available physical RAM in GB, as Task Manager's 'Available' counts it."""
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1024 ** 3


def touch(chunk):
    """Write one byte per page, so every page is resident, not just reserved.

    Why: a fresh bytearray is backed by demand-zero pages that cost nothing
    until written; untouched, the hog would hold address space, not RAM.
    """
    chunk[::PAGE] = b"\x01" * (len(chunk) // PAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leave-gb", type=float, required=True)
    ap.add_argument("--max-gb", type=float, default=24.0)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    chunks, ready, last_report = [], False, 0.0
    while True:
        avail = available_gb()
        held = len(chunks) * CHUNK / 1024 ** 3
        if avail > args.leave_gb + 0.25 and held < args.max_gb:
            chunk = bytearray(CHUNK)
            touch(chunk)
            chunks.append(chunk)
            continue
        if avail < 0.3 and chunks:
            chunks.pop()          # safety valve: never wedge the machine
        if not ready:
            ready = True
            print(f"READY held={held:.1f} GB available={avail:.2f} GB", flush=True)
        for c in chunks:          # keep all of it hot, so the OS pages others out
            touch(c)
        if time.time() - last_report > 10:
            print(f"held={held:.1f} GB available={avail:.2f} GB", flush=True)
            last_report = time.time()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
