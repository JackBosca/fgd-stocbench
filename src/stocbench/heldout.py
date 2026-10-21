"""Freeze baseline checkpoints/settings and evaluate explicit held-out partitions."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from stocbench.protocol import PARTITIONS, REPO, VARIANTS, digest, read_json, sha256, write_json

STANDARD = [f"u:{n}" for n in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400)]
DPM2 = [f"u:{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400)]
RECIPES = {
    "si": ("si", STANDARD, []),
    "fm": ("fm", STANDARD, []),
    "dpm2": ("dm", DPM2, ["model.sampler._target_=stocbench.models.diffusion.samplers.ExpSampler"]),
    "ddim": (
        "dm",
        STANDARD,
        ["model.sampler._target_=stocbench.models.diffusion.samplers.DDIMSampler", "+model.sampler.eta=0.0"],
    ),
    "ddpm": (
        "dm",
        STANDARD,
        ["model.sampler._target_=stocbench.models.diffusion.samplers.DDIMSampler", "+model.sampler.eta=1.0"],
    ),
    "cd1": ("cd", ["at:[0]"], []),
    "cd2": ("cd", ["at:[0,2]"], []),
    "addfm": ("add_fm", ["u:1"], []),
}


def lock_baselines(spec_path, output):
    """Local files only: resolve/download published files separately, then pin them."""
    spec = read_json(spec_path)
    variant = spec["variant"]
    if variant not in VARIANTS:
        raise ValueError("invalid variant")
    checkpoints = {}
    for model, entry in spec["checkpoints"].items():
        if model not in ("si", "fm", "dm", "edm", "cd", "add_fm"):
            raise ValueError(f"unknown model: {model}")
        path = Path(entry["path"]).expanduser().resolve()
        if not path.is_file() or not entry.get("provenance"):
            raise ValueError(f"missing checkpoint or provenance: {model}")
        checkpoints[model] = {**entry, "path": str(path), "sha256": sha256(path)}
    recipes = {}
    for name in spec.get("baselines", list(RECIPES)):
        model, schedules, overrides = RECIPES[name]
        if model not in checkpoints:
            raise ValueError(f"missing weights: {model}")
        teacher = {"cd": "edm", "add_fm": "fm"}.get(model)
        if teacher and teacher not in checkpoints:
            raise ValueError(f"missing teacher: {teacher}")
        extra = spec.get("model_overrides", {}).get(model, [])
        if any(not s.startswith("model.") or any(x in s for x in ("teacher_ckpt", "hf=", "hf.")) for s in extra):
            raise ValueError("model_overrides may only set model fields, excluding checkpoint/HF paths")
        schedules = spec.get("schedules", {}).get(name, schedules)
        from stocbench.utils import Schedule

        if not schedules:
            raise ValueError("empty schedule list")
        for schedule in schedules:
            Schedule.parse(schedule)
        recipes[name] = {
            "model": model,
            "teacher": teacher,
            "schedules": schedules,
            "overrides": list(overrides) + list(extra),
        }
    if not recipes:
        raise ValueError("select at least one baseline")
    lock = {
        "schema_version": 1,
        "variant": variant,
        "seed": int(spec.get("seed", 42)),
        "checkpoints": checkpoints,
        "baselines": recipes,
        "selection_provenance": spec.get("selection_provenance", "Frozen published/legacy checkpoint selection"),
    }
    lock["lock_sha256"] = digest(lock)
    write_json(output, lock)
    print(f"Checkpoint/settings lock: {output}")


def evaluation_command(manifest_path, lock_path, baseline, partition, run_dir, batch_size=32):
    from stocbench.data.heldout import load_manifest

    manifest_path, lock_path, run_dir = [Path(p).resolve() for p in (manifest_path, lock_path, run_dir)]
    m, root = load_manifest(manifest_path)
    lock = read_json(lock_path)
    if digest({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock["lock_sha256"]:
        raise ValueError("checkpoint/settings lock was modified")
    if lock["variant"] != m["variant"] or partition not in PARTITIONS:
        raise ValueError("variant/partition mismatch")
    if run_dir == REPO or REPO in run_dir.parents or run_dir.exists():
        raise ValueError("use a new run directory outside the source tree")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    selection = m["evaluation"][partition]
    recipe = lock["baselines"][baseline]
    model = recipe["model"]
    for name in (model, recipe["teacher"]):
        if name is None:
            continue
        entry = lock["checkpoints"][name]
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"checkpoint changed: {name}")
    horizon = selection["rollout_horizons"]
    if horizon != [10, 50]:
        raise ValueError("baseline wrapper expects the frozen 10/50-step protocol")
    import json

    num_cases = len(m["partitions"][partition]["cases"]) if m["variant"] == "stoc" else len(selection["stats_pairs"])
    command = [
        sys.executable,
        "-m",
        "stocbench.eval",
        f"experiment={m['variant']}",
        f"model={model}",
        f"data=heldout_{m['variant']}",
        f"data.root={root}",
        f"data.manifest={manifest_path}",
        f"data.partition={partition}",
        f"data.batch_size={batch_size}",
        "data.num_workers=0",
        f"ckpt_path={lock['checkpoints'][model]['path']}",
        f"seed={lock['seed']}",
        "evaluate=stocbench",
        f"evaluate.ensemble_size={selection['ensemble_size']}",
        f"evaluate.n_inits={num_cases}",
        f"evaluate.batch_size={batch_size}",
        "evaluate.rollout_steps=10",
        "evaluate.long_rollout_steps=50",
        f"evaluate.sampling_schedules={json.dumps(recipe['schedules'], separators=(',', ':'))}",
        f"model.name={baseline}",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.strategy=auto",
        f"hydra.run.dir={run_dir}",
        "output_dir=.",
    ]
    command += recipe["overrides"]
    if recipe["teacher"]:
        command.append(f"model.teacher_ckpt={lock['checkpoints'][recipe['teacher']]['path']}")
    evidence = {
        "manifest_sha256": sha256(manifest_path),
        "checkpoint_lock": lock,
        "partition": partition,
        "baseline": baseline,
        "command": command,
        "source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in (REPO / "src/stocbench").rglob("*.py")},
    }
    return command, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--spec", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--baseline", choices=list(RECIPES), required=True)
    run.add_argument("--partition", choices=PARTITIONS, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.command == "lock":
        lock_baselines(a.spec, a.output)
        return
    command, evidence = evaluation_command(a.manifest, a.lock, a.baseline, a.partition, a.run_dir, a.batch_size)
    print(shlex.join(command), flush=True)
    if not a.dry_run:
        from stocbench.data.heldout import verify_selection

        verify_selection(a.manifest, a.partition)
        a.run_dir.mkdir(parents=True, exist_ok=False)
        write_json(a.run_dir / "protocol_evidence.json", evidence)
        subprocess.run(command, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
