"""Plan, verify and freeze additive StocBench held-out releases; no simulation on import."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

PARTITIONS = ("validation", "test")
VARIANTS = ("stoc", "det")
REPO = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    """Exclusive creation: never replace a plan, manifest, or completed record."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def source_evidence():
    names = ("generate_holdouts.py", "heldout_engine.py", "base.py", "configs.py", "incns_stoc.py")
    paths = [REPO / "solver" / n for n in names]
    paths += [Path(__file__)]
    return {
        "commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "files": {str(p.relative_to(REPO)): sha256(p) for p in paths},
    }


def stream_key(config, variant, partition, kind, identity, phase, member=0):
    """Stable 64-bit JAX key data, independent of job layout and batch size.

    The high bit distinguishes root keys from legacy PRNGKey(integer) roots.
    SHA-256 derives streams; full preimages are retained through the plan/IDs.
    """
    tag = [config["rng_namespace"], variant, partition, kind, int(identity), phase, int(member)]
    raw = hashlib.sha256(json.dumps(tag, separators=(",", ":")).encode()).digest()
    return np.array([int.from_bytes(raw[:4], "big") | (1 << 31), int.from_bytes(raw[4:8], "big")], dtype=np.uint32)


def validate_config(c):
    if c["schema_version"] != 1 or not c["name"] or not c["rng_namespace"]:
        raise ValueError("invalid protocol identity")
    for key in ("grid", "warmup_frames", "trajectory_frames", "trajectories_per_partition", "reference_members"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if set(c["partitions"]) != set(PARTITIONS):
        raise ValueError("exactly validation and test partitions are required")
    if c["solver"]["n"] % c["grid"] or c["solver"]["std"] <= 0:
        raise ValueError("invalid grid reduction or normalization")
    ratio = c["solver"]["sample_dt"] / c["solver"]["dt"]
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError("snapshot interval must be a positive integer number of solver steps")
    if not c["rollout_horizons"] or any(h < 1 for h in c["rollout_horizons"]):
        raise ValueError("positive rollout horizons required")
    if c["rollout_start"] < 0 or c["rollout_start"] + max(c["rollout_horizons"]) >= c["trajectory_frames"]:
        raise ValueError("rollout outside trajectory")
    seeds, cases = set(), set()
    for partition in PARTITIONS:
        p = c["partitions"][partition]
        if p["trajectory_seed"] in seeds | {42, 43, 44, 45}:
            raise ValueError("trajectory batch IDs must be new and disjoint")
        seeds.add(p["trajectory_seed"])
        if not 1 <= p["cases"] <= c["trajectories_per_partition"]:
            raise ValueError("one-step conditions must use distinct trajectories")
        ids = set(range(p["case_start"], p["case_start"] + p["cases"]))
        if ids & (cases | set(range(100, 148))):
            raise ValueError("case IDs must be new and disjoint")
        cases |= ids
        if any(c["generated_members"][partition][v] < 1 for v in VARIANTS):
            raise ValueError("positive generated ensemble sizes required")


def load_plan(release):
    plan = read_json(Path(release) / "plan.json")
    validate_config(plan["config"])
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("plan checksum mismatch")
    return plan


def plan_release(config, legacy_root, release):
    validate_config(config)
    release, legacy_root = Path(release).resolve(), Path(legacy_root).resolve()
    if release in (legacy_root, REPO) or legacy_root in release.parents or REPO in release.parents:
        raise ValueError("release must be outside the source tree and legacy data root")
    training = {}
    for variant in VARIANTS:
        training[variant] = []
        for seed in (42, 43, 44):
            p = legacy_root / f"incns_{variant}/64/traj_seed_{seed}.npy"
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            expected = (500, 200, 1 if variant == "stoc" else 2, 64, 64)
            if a.shape != expected or a.dtype != np.float32:
                raise ValueError(f"unexpected original training array: {p}: {a.shape}, {a.dtype}")
            if config["grid"] != 64:
                raise ValueError("production plans preserve the original 64x64 training grid")
            training[variant].append({"seed": seed, "path": str(p), "shape": list(a.shape)})
    plan = {"schema_version": 1, "config": config, "training": training, "generator": source_evidence()}
    plan["plan_sha256"] = digest(plan)
    release.mkdir(parents=True, exist_ok=False)
    write_json(release / "plan.json", plan)
    print(f"Plan: {release / 'plan.json'}", flush=True)
    return plan


def item_path(release, variant, partition, kind, identity):
    return Path(release) / "generated" / variant / partition / f"{kind}-{identity:06d}"


def verify_item(path, plan, variant, partition, kind, identity):
    record = read_json(Path(path) / "record.json")
    expected = [variant, partition, kind, identity]
    if record["identity"] != expected or record["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError(f"wrong provenance: {path}")
    if record["generator_files"] != plan["generator"]["files"]:
        raise ValueError(f"generator differs from plan: {path}")
    required = {"trajectory.npy"} if kind == "trajectory" else {"init.npy", "init_highres.npy", "raw.npy"}
    if set(record["files"]) != required:
        raise ValueError(f"unexpected artifact inventory: {path}")
    for name, checksum in record["files"].items():
        if sha256(Path(path) / name) != checksum:
            raise ValueError(f"artifact checksum mismatch: {path}/{name}")
    return record


def check_array(path, shape):
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    if a.shape != tuple(shape) or a.dtype != np.float32:
        raise ValueError(f"invalid array {path}: {a.shape}, {a.dtype}; expected {shape}, float32")
    for chunk in a:
        if not np.isfinite(chunk).all():
            raise ValueError(f"nonfinite values: {path}")
    return a


def check_cross_partition(fields, partition, seen):
    """Check exact vorticity duplicates across partitions, including reference futures."""
    for field in fields:
        key = hashlib.sha256(np.ascontiguousarray(field).tobytes()).digest()
        previous = seen.setdefault(key, partition)
        if previous != partition:
            raise ValueError(f"identical vorticity fields in {previous} and {partition}")


def spectrum_numpy(fields):
    """Same radial FFT normalization/bins as stocbench.metrics.radial_enstrophy."""
    n, m = fields.shape[-2:]
    yy, xx = np.meshgrid(np.arange(n) - n // 2, np.arange(m) - m // 2, indexing="ij")
    bins = np.floor(np.sqrt(xx**2 + yy**2)).astype(int).ravel()
    power = np.abs(np.fft.fftshift(np.fft.fft2(fields), axes=(-2, -1))) ** 2 / (n * m)
    return np.stack([np.bincount(bins, weights=f.ravel(), minlength=bins.max() + 1) for f in power.reshape(-1, n, m)])


def freeze_release(release):
    """Full checks and assembly; run as a CPU job. Publish frozen/ atomically."""
    release = Path(release).resolve()
    plan = load_plan(release)
    c = plan["config"]
    destination = release / "frozen"
    if destination.exists():
        raise FileExistsError(f"already frozen: {destination}; use verify")
    # Two concurrent freezes must not build/publish competing releases.
    lock = release / ".freeze.lock"
    lock.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=".freeze-", dir=release) as temp:
            out = Path(temp)
            generation_records = []
            for variant in VARIANTS:
                root = out / f"incns_{variant}/64"
                root.mkdir(parents=True)
                channels = 1 if variant == "stoc" else 2
                seen = {}
                train_groups = []
                for entry in plan["training"][variant]:
                    source = Path(entry["path"])
                    a = check_array(source, entry["shape"])
                    for trajectory in a:
                        check_cross_partition(trajectory[:, 0], "train", seen)
                    (root / f"traj_seed_{entry['seed']}.npy").symlink_to(source)
                    train_groups.append({"seed": entry["seed"], "ids": list(range(a.shape[0]))})
                manifest = {
                    "schema_version": 1,
                    "name": c["name"],
                    "variant": variant,
                    "plan_sha256": plan["plan_sha256"],
                    "partitions": {"train": {"trajectories": train_groups, "cases": []}},
                    "evaluation": {},
                }
                for partition in PARTITIONS:
                    p = c["partitions"][partition]
                    count, frames, grid = c["trajectories_per_partition"], c["trajectory_frames"], c["grid"]
                    shape = (count, frames, channels, grid, grid)
                    combined = np.lib.format.open_memmap(
                        root / f"traj_seed_{p['trajectory_seed']}.npy", mode="w+", dtype=np.float32, shape=shape
                    )
                    spec_sum = None
                    start, horizon = c["rollout_start"], max(c["rollout_horizons"])
                    for i in range(count):
                        directory = item_path(release, variant, partition, "trajectory", i)
                        generation_records.append(verify_item(directory, plan, variant, partition, "trajectory", i))
                        a = check_array(directory / "trajectory.npy", shape[1:])
                        check_cross_partition(a[:, 0], partition, seen)
                        combined[i] = a
                        spectrum = spectrum_numpy(a[start + 1 : start + horizon + 1, 0])
                        spec_sum = spectrum if spec_sum is None else spec_sum + spectrum
                        print(f"verify {variant}/{partition} trajectory={i + 1}/{count}", flush=True)
                    combined.flush()
                    del combined
                    spec_name = f"enstrophy_{partition}.npy"
                    np.save(root / spec_name, (spec_sum / count).astype(np.float32))
                    cases = list(range(p["case_start"], p["case_start"] + p["cases"])) if variant == "stoc" else []
                    for case in cases:
                        directory = item_path(release, variant, partition, "case", case)
                        generation_records.append(verify_item(directory, plan, variant, partition, "case", case))
                        init = check_array(directory / "init.npy", (1, grid, grid))
                        high = check_array(directory / "init_highres.npy", (1, c["solver"]["n"], c["solver"]["n"]))
                        factor = c["solver"]["n"] // grid
                        if not np.allclose(
                            high.reshape(1, grid, factor, grid, factor).mean((2, 4)), init, rtol=2e-5, atol=2e-6
                        ):
                            raise ValueError(f"high/coarse condition mismatch: {directory}")
                        raw = check_array(directory / "raw.npy", (c["reference_members"], 1, 1, grid, grid))
                        check_cross_partition(init, partition, seen)
                        check_cross_partition(raw[:, 0, 0], partition, seen)
                        conditions = root / "conditions"
                        conditions.mkdir(exist_ok=True)
                        shutil.copyfile(directory / "init_highres.npy", conditions / f"step_seed_{case}.npy")
                        # Streaming zip of NPYs avoids expanding a 5000-member ensemble in RAM.
                        with zipfile.ZipFile(root / f"step_seed_{case}.npz", "x") as archive:
                            for name in ("init.npy", "raw.npy"):
                                archive.write(directory / name, name)
                            mean = raw[:, 0].mean(0, dtype=np.float64).astype(np.float32)
                            std = raw[:, 0].std(0, dtype=np.float64).astype(np.float32)
                            for name, values in (("mean.npy", mean), ("std.npy", std)):
                                with archive.open(name, "w", force_zip64=True) as stream:
                                    np.save(stream, values, allow_pickle=False)
                        print(f"verify {variant}/{partition} case={case}", flush=True)
                    groups = [{"seed": p["trajectory_seed"], "ids": list(range(count))}]
                    manifest["partitions"][partition] = {"trajectories": groups, "cases": cases}
                    manifest["evaluation"][partition] = {
                        "stats_pairs": [[p["trajectory_seed"], i, 0] for i in range(p["cases"])],
                        "rollout_starts": [[p["trajectory_seed"], i, start] for i in range(count)],
                        "rollout_horizons": c["rollout_horizons"],
                        "reference_spectrum": spec_name,
                        "ensemble_size": c["generated_members"][partition][variant],
                    }
                write_json(out / f"{variant}.json", manifest)
            write_json(out / "plan.json", plan)
            write_json(out / "generation_records.json", generation_records)
            files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
            write_json(out / "release.json", {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "files": files})
            os.rename(out, destination)
        print(f"Frozen release: {destination}", flush=True)
    finally:
        lock.rmdir()


def verify_frozen(root):
    root = Path(root).resolve()
    record = read_json(root / "release.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "release.json"}
    if actual != set(record["files"]):
        raise ValueError("frozen file inventory changed")
    for name, checksum in record["files"].items():
        if sha256(root / name) != checksum:
            raise ValueError(f"changed frozen artifact: {name}")
        print(f"verified {name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=REPO / "protocols/cfgd-stocbench-v1.json")
    plan.add_argument("--legacy-root", type=Path, required=True)
    plan.add_argument("--release", type=Path, required=True)
    for name in ("freeze", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--release", type=Path, required=True)
    a = p.parse_args()
    if a.command == "plan":
        plan_release(read_json(a.config), a.legacy_root, a.release)
    elif a.command == "freeze":
        freeze_release(a.release)
    else:
        verify_frozen(a.release / "frozen")


if __name__ == "__main__":
    main()
