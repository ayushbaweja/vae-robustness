#!/usr/bin/env python3
"""4-GPU parallel orchestrator for the VAE robustness sweep.

Each (model, ε, loss) experiment runs as a single subprocess pinned to one GPU
via CUDA_VISIBLE_DEVICES. A queue of GPU slots throttles concurrency to
`--num-gpus` (default 4). Existing summary.json files cause the experiment to
be skipped, matching run_sweep.sh semantics.

Usage:
    python run_sweep_parallel.py \
        --input-dir resources/test-images-imagenet25 \
        --output-root results/imagenet25
    python run_sweep_parallel.py --models sd15 flux1
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# (display name -> (script, output subdir under output_root))
MODELS = {
    "sd15":      ("pgd_sd15_vae.py",      "sd15_pgd"),
    "flux1":     ("pgd_flux_vae.py",      "flux1_pgd"),
    "flux2":     ("pgd_flux2_vae.py",     "flux2_pgd"),
    "cogvideox": ("pgd_cogvideox_vae.py", "cogvideox_pgd"),
    "ltx":       ("pgd_ltx_vae.py",       "ltx_pgd"),
}

EPSILONS = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]
LOSSES = ["pixel", "latent"]

ALPHA = {0.02: 0.005, 0.04: 0.007, 0.06: 0.01, 0.1: 0.015, 0.15: 0.02, 0.2: 0.02}
ITERS = {0.02: 40,    0.04: 40,    0.06: 40,   0.1: 40,    0.15: 50,   0.2: 60}


@dataclass
class Job:
    model: str
    eps: float
    loss: str
    script: str
    output_dir: Path
    input_dir: Path

    @property
    def result_dir(self) -> Path:
        return self.output_dir / f"eps_{self.eps}_{self.loss}"

    @property
    def already_done(self) -> bool:
        return (self.result_dir / "summary.json").exists()

    def cmd(self, python_exe: str) -> list[str]:
        return [
            python_exe, self.script,
            "--input_dir",  str(self.input_dir),
            "--output_dir", str(self.output_dir),
            "--epsilon",    str(self.eps),
            "--alpha",      str(ALPHA[self.eps]),
            "--num_iter",   str(ITERS[self.eps]),
            "--loss",       self.loss,
        ]


# ── Console with locking so concurrent prints don't interleave ──────────────
_print_lock = threading.Lock()
def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_job(job: Job, gpu_q: "queue.Queue[int]", logs_dir: Path,
            python_exe: str) -> tuple[Job, str, int, float]:
    if job.already_done:
        return job, "skip", 0, 0.0

    gpu = gpu_q.get()
    t0 = time.time()
    log(f"[GPU {gpu}] START  {job.model:>9s}  eps={job.eps:<5}  loss={job.loss}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Avoid HF cache thrash: each process can share the cache dir.
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("DIFFUSERS_VERBOSITY",   "error")

    log_path = logs_dir / f"{job.model}_eps{job.eps}_{job.loss}_gpu{gpu}.log"
    try:
        with open(log_path, "wb") as lf:
            res = subprocess.run(job.cmd(python_exe), env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        rc = res.returncode
    finally:
        gpu_q.put(gpu)

    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAIL(rc={rc})"
    log(f"[GPU {gpu}] {status:<7} {job.model:>9s}  eps={job.eps:<5}  loss={job.loss}  "
        f"({dt/60:.1f} min)  log={log_path}")
    return job, status, rc, dt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir",   default="resources/test-images-imagenet25",
                    help="Directory of input images.")
    ap.add_argument("--output-root", default="results/imagenet25",
                    help="Root directory under which per-model results are written.")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()),
                    help="Subset of models to run.")
    ap.add_argument("--num-gpus", type=int, default=4,
                    help="Number of concurrent GPU workers.")
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids to use (e.g. '0,1,2,3'). "
                         "Overrides --num-gpus.")
    ap.add_argument("--python", default=str(Path(".venv/bin/python").absolute()),
                    help="Python interpreter used for subprocesses "
                         "(defaults to ./.venv/bin/python so the project's locked "
                         "environment is used). Symlinks are NOT followed — calling "
                         "the venv launcher path is required for the venv's "
                         "site-packages to be picked up.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the job plan and exit.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    input_dir   = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not Path(args.python).exists():
        print(f"ERROR: python interpreter not found: {args.python}", file=sys.stderr)
        print("       Pass --python /path/to/python or run `uv sync`.", file=sys.stderr)
        return 2

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
    else:
        gpus = list(range(args.num_gpus))

    # Build job list
    jobs: list[Job] = []
    for m in args.models:
        script, subdir = MODELS[m]
        out = output_root / subdir
        for eps in EPSILONS:
            for loss in LOSSES:
                jobs.append(Job(model=m, eps=eps, loss=loss,
                                script=script, output_dir=out,
                                input_dir=input_dir))

    pending = [j for j in jobs if not j.already_done]
    skipped = [j for j in jobs if j.already_done]

    print("="*72)
    print(f" Sweep plan: input={input_dir.name}  output_root={output_root}")
    print(f"  Total jobs:  {len(jobs)}")
    print(f"  To run:      {len(pending)}")
    print(f"  Already done:{len(skipped)}")
    print(f"  GPUs:        {gpus}")
    print("="*72)
    if args.dry_run or not pending:
        for j in pending:
            print(f"  RUN  {j.model:>9s}  eps={j.eps:<5}  loss={j.loss}  -> {j.result_dir}")
        return 0

    logs_dir = output_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    t_start = time.time()
    failures: list[Job] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(run_job, j, gpu_q, logs_dir, args.python)
                   for j in pending]
        for fut in as_completed(futures):
            job, status, rc, _ = fut.result()
            if rc != 0:
                failures.append(job)

    dt = time.time() - t_start
    print("="*72)
    print(f"  Done. Wall time: {dt/60:.1f} min   "
          f"ran={len(pending)-len(failures)}  failed={len(failures)}  skipped={len(skipped)}")
    if failures:
        print("  Failed jobs:")
        for j in failures:
            print(f"    {j.model}  eps={j.eps}  loss={j.loss}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
