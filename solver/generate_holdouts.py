"""Generate resumable, independently keyed holdout shards from a frozen plan."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from stocbench.protocol import (
    PARTITIONS,
    VARIANTS,
    item_path,
    load_plan,
    sha256,
    source_evidence,
    stream_key,
    verify_item,
    write_json,
)


def save_record(directory, plan, variant, partition, kind, identity, runtime):
    files = {p.name: sha256(p) for p in sorted(directory.glob("*.npy"))}
    write_json(
        directory / "record.json",
        {
            "identity": [variant, partition, kind, identity],
            "plan_sha256": plan["plan_sha256"],
            "generator_files": plan["generator"]["files"],
            "files": files,
            "runtime": runtime,
            "rng": "stream_key(namespace, variant, partition, kind, identity, phase, member)",
        },
    )


def generate(release, variant, partition, kind, shard_index=0, shard_count=1, batch_size=8, member_batch=32):
    import jax

    from solver.heldout_engine import HeldoutEngine

    release = Path(release).resolve()
    plan = load_plan(release)
    if source_evidence()["files"] != plan["generator"]["files"]:
        raise ValueError("generator source changed since planning; restore it or create a new release")
    if (release / "frozen").exists():
        raise ValueError("release is already frozen")
    if not 0 <= shard_index < shard_count or min(batch_size, member_batch) < 1:
        raise ValueError("invalid shard/batch arguments")
    if kind == "case" and variant != "stoc":
        raise ValueError("deterministic cases come from trajectory pairs")
    c = plan["config"]
    p = c["partitions"][partition]
    ids = (
        list(range(c["trajectories_per_partition"]))
        if kind == "trajectory"
        else list(range(p["case_start"], p["case_start"] + p["cases"]))
    )
    assigned = ids[shard_index::shard_count]
    pending = []
    for identity in assigned:
        path = item_path(release, variant, partition, kind, identity)
        if path.exists():
            verify_item(path, plan, variant, partition, kind, identity)
            print(f"already complete {variant}/{partition}/{kind}/{identity}", flush=True)
        else:
            pending.append(identity)
    if not pending:
        print("All assigned items already verified.", flush=True)
        return
    engine = HeldoutEngine(c, variant)
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "devices": [str(d) for d in jax.devices()],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "batch_size": batch_size,
        "member_batch": member_batch,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    parent = item_path(release, variant, partition, kind, pending[0]).parent
    parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(f"{variant}/{partition}/{kind}: {message}", flush=True)

    if kind == "trajectory":
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset : offset + batch_size]
            log(f"generating IDs {chunk}")
            warm = np.stack([stream_key(c, variant, partition, kind, i, "warmup") for i in chunk])
            keys = np.stack([stream_key(c, variant, partition, kind, i, "trajectory") for i in chunk])
            trajectories = engine.trajectories(warm, keys, log)
            for identity, trajectory in zip(chunk, trajectories, strict=True):
                target = item_path(release, variant, partition, kind, identity)
                with tempfile.TemporaryDirectory(prefix=".pending-", dir=parent) as temp:
                    directory = Path(temp)
                    np.save(directory / "trajectory.npy", trajectory)
                    save_record(directory, plan, variant, partition, kind, identity, runtime)
                    os.rename(directory, target)
                log(f"complete ID={identity}")
    else:
        for identity in pending:
            target = item_path(release, variant, partition, kind, identity)
            log(f"generating ID={identity}")
            high = engine.condition(
                stream_key(c, variant, partition, kind, identity, "warmup"),
                stream_key(c, variant, partition, kind, identity, "condition"),
            )
            with tempfile.TemporaryDirectory(prefix=".pending-", dir=parent) as temp:
                directory = Path(temp)
                np.save(directory / "init_highres.npy", high)
                factor, grid = c["solver"]["n"] // c["grid"], c["grid"]
                np.save(directory / "init.npy", high.reshape(1, grid, factor, grid, factor).mean((2, 4)))
                raw = np.lib.format.open_memmap(
                    directory / "raw.npy", mode="w+", dtype=np.float32, shape=(c["reference_members"], 1, 1, grid, grid)
                )
                for offset in range(0, c["reference_members"], member_batch):
                    end = min(offset + member_batch, c["reference_members"])
                    keys = np.stack(
                        [
                            stream_key(c, variant, partition, kind, identity, "future", member)
                            for member in range(offset, end)
                        ]
                    )
                    raw[offset:end, 0, 0] = engine.futures(high, keys)
                    log(f"ID={identity} futures={end}/{c['reference_members']}")
                raw.flush()
                del raw
                save_record(directory, plan, variant, partition, kind, identity, runtime)
                os.rename(directory, target)
            log(f"complete ID={identity}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--partition", choices=PARTITIONS, required=True)
    p.add_argument("--kind", choices=("trajectory", "case"), required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--member-batch", type=int, default=32)
    a = p.parse_args()
    generate(**vars(a))


if __name__ == "__main__":
    main()
