r"""Hold a model resident on one device, do nothing to it for hours, then generate.

Why: issue #38 reports that an Arc iGPU running with ``--idle-timeout 0`` --
so nothing ever unloads or touches the pipeline -- answers a short prompt with
``CL_OUT_OF_RESOURCES`` after two to three hours of sitting idle, and does not
recover. The reporter's own control is the tell: with the idle watchdog left
at its default the unload/reload cycle happens and he sees no crash. That
makes *elapsed idle time with a live allocation* the variable, and nothing in
the repo tests it -- the residency evidence behind ``_keepalive_eligible``
covers 300 seconds on an iGPU, not 3 hours.

This runs the bare ``openvino_genai`` pipeline, no NoLlama, per the standing
order in CLAUDE.md: if a bare pipeline dies the same way, the server cannot be
the cause and the finding belongs upstream.

    venv\Scripts\python scripts\idle-residency-probe.py ~\models\some-int4-ov
    venv\Scripts\python scripts\idle-residency-probe.py <dir> --rungs 5,45,120,200
    venv\Scripts\python scripts\idle-residency-probe.py <dir> --device CPU

Out: a banner of stack/driver/budget provenance, one block per idle rung, and
a verdict line. Always exits 0 -- a failing rung is the result, not a broken
run.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import openvino as ov
import openvino_genai as ovg
import psutil

# Idle minutes per rung. Each rung's clock starts after the previous rung's
# generate returns, so a rung IS an uninterrupted idle window of that length.
# The default reaches past the 2-3h the #38 reporter needed, with earlier
# rungs to locate a threshold rather than only prove one exists.
DEFAULT_RUNGS = "5,45,120,200"

# Short on purpose. A long prompt would let a prefill-time allocation failure
# masquerade as the idle effect; #38's crash arrived on "two short messages".
PROMPT = "Name three primary colours."
MAX_TOKENS = 24


def _now():
    """Wall-clock stamp for the log, local time to match console output."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def host_mem():
    """Free and total physical RAM in GB.

    Why this and not a VRAM counter on an integrated GPU: a 140V's "VRAM" is
    host RAM, so the number that moves when an allocation lives or dies is
    this one. Shared GPU memory is a ceiling, not a carve-out.

    In: nothing. Out: (free_gb, total_gb) floats.
    """
    vm = psutil.virtual_memory()
    return vm.available / 2**30, vm.total / 2**30


def proc_mem():
    """This process's commit and working set, in GB, in that order.

    Why both, and why commit is the one to read: on an integrated GPU the
    weights are ordinary host pages, so Windows can trim them out of the
    working set, and a reader watching only that column will call it an
    unload. Commit is what says the allocation is still alive.

    [OBSERVED 2026-09-15, 140V, Qwen3-8B-int4, driver 32.0.101.8991] Two runs
    on the same box disagreed, and the disagreement is the finding. With other
    work competing for RAM, a 45-minute idle took the working set from 5.06 to
    0.16 GB while commit held at 5.22 GB; the next generate still returned in
    1.29s against a 1.55s cold baseline, so the pages came back off the
    standby list, not off disk. On a quieter run the working set held flat at
    4.96 GB across 210 minutes of the same idle, 211 samples, never trimmed.
    So the trim tracks memory pressure, not elapsed idle -- do not read a
    falling working set as an idle timer.

    In: nothing. Out: (commit_gb, working_set_gb) floats, for this process
    only, so another model server on the box cannot contaminate the reading.
    """
    mi = psutil.Process().memory_info()
    # .private is Windows-only; off Windows the pair degenerates to rss twice
    # rather than ending a six-hour run on an AttributeError.
    return getattr(mi, "private", mi.rss) / 2**30, mi.rss / 2**30


def gpu_driver_version():
    """The Intel display driver version, or None when it cannot be read.

    Why it is in the banner rather than left to the writer to remember: every
    measurement in this repo is recorded against a driver, because "update
    your driver" is the first thing Intel says to a report, and a number with
    no driver beside it cannot answer that.

    In: nothing. Out: version string, or None on any failure -- a missing
    driver line must never abort a six-hour run.
    """
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             # Match on the vendor, not on "Arc": the 285K's iGPU enumerates
             # as the bare "Intel(R) Graphics" and an Arc-only filter left the
             # banner reading "unreadable" on the one box whose driver
             # mattered most [OBSERVED 2026-09-16]. Vendor-matching also keeps
             # the RX 580, the RTX 5090 and TeamViewer's virtual adapter out.
             "(Get-CimInstance Win32_VideoController | "
             "Where-Object { $_.Name -match 'Intel' } | "
             "Select-Object -First 1 -ExpandProperty DriverVersion)"],
            capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip() or None
    except Exception:
        return None


def host_uptime_h():
    """Hours since boot, or None on failure.

    Why it is recorded: the #38 reporter's standing hypothesis is that his
    crash tracks machine uptime rather than anything NoLlama does, and a run
    that does not write down its own uptime cannot speak to that either way.

    In: nothing. Out: float hours, or None.
    """
    try:
        return (time.time() - psutil.boot_time()) / 3600.0
    except Exception:
        return None


def banner(model_dir, device):
    """Print everything a reader needs to re-run this, before anything loads.

    Why first: if the load itself dies, the provenance is already on screen.

    In: model directory and device string. Out: None; prints only.
    """
    free, total = host_mem()
    core = ov.Core()
    print(f"openvino {ov.__version__}")
    print(f"genai    {ovg.__version__}")
    print(f"python   {sys.version.split()[0]}")
    print(f"model    {model_dir}")
    print(f"device   {device}")
    if device.startswith("GPU"):
        for prop in ("FULL_DEVICE_NAME", "GPU_DEVICE_TOTAL_MEM_SIZE"):
            try:
                print(f"  {prop:26} {core.get_property(device, prop)}")
            except Exception as e:
                print(f"  {prop:26} unreadable ({e})")
    print(f"  {'driver':26} {gpu_driver_version() or 'unreadable'}")
    up = host_uptime_h()
    print(f"  {'host uptime at start':26} "
          f"{('%.1f h' % up) if up is not None else 'unreadable'}")
    print(f"  {'host RAM':26} {free:.1f} GB free of {total:.1f} GB")
    print()


def generate_once(pipe, cfg, label):
    """One short generate, timed, with any exception captured rather than raised.

    Why it must not raise: the whole point of a rung is to learn what the
    device does after idling, including dying. A traceback that ends the
    process throws away every later rung and the recovery probe below it.

    In: a live pipeline, a GenerationConfig, and a label for the log. Out: a
    dict with ok/seconds and either text or error; never raises for a generate
    failure.
    """
    t0 = time.time()
    try:
        text = pipe.generate(PROMPT, cfg)
        dt = time.time() - t0
        print(f"  {_now()} {label}: OK in {dt:.2f}s -- "
              f"{str(text)[:60].strip()!r}", flush=True)
        return {"ok": True, "seconds": dt, "text": str(text)}
    except Exception as e:
        dt = time.time() - t0
        print(f"  {_now()} {label}: FAILED after {dt:.2f}s", flush=True)
        for line in str(e).splitlines():
            print(f"      {line}", flush=True)
        return {"ok": False, "seconds": dt, "error": str(e),
                "traceback": traceback.format_exc()}


def idle(minutes, sample_sec):
    """Sleep for `minutes`, sampling host memory but never touching the device.

    Why the sampler is deliberately blind to the GPU: an OpenCL query during
    the idle window would be exactly the keepalive touch whose absence is
    under test. Host-side counters are free to read and, on an integrated GPU
    whose "VRAM" is host RAM, are the residency signal anyway -- see proc_mem
    for why the commit column and not the working set is the one that answers
    "is the allocation still alive".

    In: idle minutes and a sampling period in seconds. Out: a list of
    (elapsed_seconds, free_gb, commit_gb, working_set_gb) samples, one per
    period plus a final one at the end of the window.
    """
    end = time.time() + minutes * 60
    started = time.time()
    samples = []
    while time.time() < end:
        free, _ = host_mem()
        commit, ws = proc_mem()
        samples.append((round(time.time() - started), round(free, 2),
                        round(commit, 2), round(ws, 2)))
        time.sleep(min(sample_sec, max(1, end - time.time())))
    free, _ = host_mem()
    commit, ws = proc_mem()
    samples.append((round(time.time() - started), round(free, 2),
                    round(commit, 2), round(ws, 2)))
    return samples


def _write(path, record, quiet=False):
    """Persist the record, if a path was given, without letting IO end the run.

    Why it is called after every rung and not only at the end: this script's
    whole cost is wall-clock, so a crash on the last rung must not take the
    earlier ones with it. Each call rewrites the file whole, so the last write
    wins and a partial file is still a valid record of everything finished.

    In: an optional path, the record dict, and quiet to suppress the progress
    line for the per-rung flushes. Out: None; prints where it went, or why it
    could not.
    """
    if not path:
        return
    record["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["uptime_h_at_end"] = host_uptime_h()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        if not quiet:
            print(f"wrote {path}", flush=True)
    except Exception as e:
        print(f"could not write {path}: {e}", flush=True)


def main():
    """Load once, then walk the idle rungs, reporting the first that fails.

    Why it probes again after a failure: "does it recover" is half the claim in
    #38 -- the reporter had to restart NoLlama, and sometimes reboot. So a
    failed rung is immediately retried twice, which separates a poisoned
    context from a one-off allocation refusal.

    In: argv. Out: exit code 0 always; the verdict is in the text and the
    --json file, so a caller reads those rather than the status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Model directory (an OpenVINO export)")
    ap.add_argument("--device", default="GPU", help="GPU, CPU or NPU (default GPU)")
    ap.add_argument("--rungs", default=DEFAULT_RUNGS,
                    help=f"Comma-separated idle minutes (default {DEFAULT_RUNGS})")
    ap.add_argument("--sample-sec", type=int, default=60,
                    help="Host-memory sampling period during idle (default 60)")
    ap.add_argument("--json", help="Write the full result record here")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    rungs = [float(x) for x in args.rungs.split(",") if x.strip()]

    banner(model_dir, args.device)

    record = {"model_dir": model_dir, "device": args.device,
              "openvino": ov.__version__, "genai": ovg.__version__,
              "driver": gpu_driver_version(),
              "uptime_h_at_start": host_uptime_h(),
              "started": datetime.datetime.now().isoformat(timespec="seconds"),
              "rungs": []}

    print(f"{_now()} loading on {args.device} ...", flush=True)
    t0 = time.time()
    pipe = ovg.LLMPipeline(model_dir, args.device)
    load_s = time.time() - t0
    free, _ = host_mem()
    commit, ws = proc_mem()
    print(f"{_now()} loaded in {load_s:.1f}s -- commit {commit:.2f} GB, "
          f"ws {ws:.2f} GB, host free {free:.1f} GB\n", flush=True)
    record["load_seconds"] = load_s

    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = MAX_TOKENS
    # No penalties: the Phi-3.5-vision lesson is that our own defaults can be
    # the bug, and this probe is about the device, not about sampling.

    print("rung 0 (warm baseline, no idle)", flush=True)
    base = generate_once(pipe, cfg, "generate")
    record["baseline"] = base
    if not base["ok"]:
        print("\nVERDICT: failed before any idle -- this is not the idle effect.")
        _write(args.json, record)
        return 0

    for i, minutes in enumerate(rungs, start=1):
        back = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        print(f"\nrung {i}: idle {minutes:g} min (no device call until it ends) "
              f"-- back at {back.strftime('%H:%M:%S')}", flush=True)
        samples = idle(minutes, args.sample_sec)
        free, _ = host_mem()
        commit, ws = proc_mem()
        print(f"  {_now()} idle over -- host free {free:.1f} GB, commit "
              f"{commit:.2f} GB, ws {ws:.2f} GB (at rung start: free "
              f"{samples[0][1]:.2f}, commit {samples[0][2]:.2f}, ws "
              f"{samples[0][3]:.2f})", flush=True)
        res = generate_once(pipe, cfg, f"generate after {minutes:g} min idle")
        rung = {"idle_minutes": minutes, "result": res, "samples": samples}

        if not res["ok"]:
            rung["retries"] = [generate_once(pipe, cfg, f"  retry {n}")
                               for n in (1, 2)]
            record["rungs"].append(rung)
            recovered = any(r["ok"] for r in rung["retries"])
            print(f"\nVERDICT: failed after {minutes:g} min idle; "
                  f"{'RECOVERED on retry' if recovered else 'did NOT recover'}.")
            _write(args.json, record)
            return 0

        record["rungs"].append(rung)
        # Flush after every rung, not once at the end. A run that reaches the
        # last rung has already spent hours holding a multi-GB allocation,
        # which is exactly when something else on the box gets the process
        # killed. [OBSERVED 2026-09-15] The first 140V run died four minutes
        # short of its 200-minute rung and wrote no JSON at all, so three good
        # rungs survived only as console text.
        _write(args.json, record, quiet=True)

    total = sum(rungs)
    print(f"\nVERDICT: survived every rung -- {total:g} min of idle in "
          f"{len(rungs)} windows, longest {max(rungs):g} min, no failure.")
    _write(args.json, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
