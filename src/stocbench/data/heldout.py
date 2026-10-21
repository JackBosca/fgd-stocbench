"""Datasets with immutable case/trajectory membership and explicit evaluation starts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stocbench.protocol import read_json, sha256

from . import IncnsData, TrajTrain
from .stoc import StocStats


def load_manifest(path, root=None):
    path = Path(path).resolve()
    release = read_json(path.parent / "release.json")
    if release["files"].get(path.name) != sha256(path):
        raise ValueError("manifest is not part of the frozen release")
    m = read_json(path)
    if m["schema_version"] != 1 or m["variant"] not in ("stoc", "det"):
        raise ValueError("invalid manifest")
    if m["plan_sha256"] != release["plan_sha256"]:
        raise ValueError("manifest/release plan mismatch")
    expected_root = path.parent / f"incns_{m['variant']}/64"
    if root is not None and Path(root).resolve() != expected_root.resolve():
        raise ValueError("data root does not belong to this frozen manifest")
    trajectories, cases = set(), set()
    for split in ("train", "validation", "test"):
        members = m["partitions"][split]
        ids = [(group["seed"], i) for group in members["trajectories"] for i in group["ids"]]
        if len(set(ids)) != len(ids) or trajectories.intersection(ids):
            raise ValueError("overlapping trajectories")
        trajectories.update(ids)
        if len(set(members["cases"])) != len(members["cases"]) or cases.intersection(members["cases"]):
            raise ValueError("overlapping cases")
        cases.update(members["cases"])
        if split == "train":
            if members["cases"]:
                raise ValueError("reference cases cannot be training data")
            continue
        for key in ("stats_pairs", "rollout_starts"):
            starts = m["evaluation"][split][key]
            if not starts or len({tuple(s) for s in starts}) != len(starts):
                raise ValueError("empty/duplicate evaluation starts")
            if any((seed, i) not in set(ids) or frame < 0 for seed, i, frame in starts):
                raise ValueError("evaluation start outside its assigned partition")
    return m, expected_root


def verify_selection(manifest_path, partition):
    """Hash only the held-out files consumed by this evaluation, before inference."""
    m, root = load_manifest(manifest_path)
    selection = m["partitions"][partition]
    names = {f"traj_seed_{group['seed']}.npy" for group in selection["trajectories"]}
    names.update(f"step_seed_{case}.npz" for case in selection["cases"])
    names.add(m["evaluation"][partition]["reference_spectrum"])
    release = read_json(Path(manifest_path).resolve().parent / "release.json")
    for name in sorted(names):
        relative = f"incns_{m['variant']}/64/{name}"
        if sha256(root / name) != release["files"].get(relative):
            raise ValueError(f"held-out artifact changed: {relative}")


class SelectedWindows(Dataset):
    def __init__(self, root, starts, horizon):
        self.starts = starts
        self.horizon = int(horizon)
        self.arrays = {
            seed: np.load(Path(root) / f"traj_seed_{seed}.npy", mmap_mode="r", allow_pickle=False)
            for seed, _, _ in starts
        }
        for seed, i, frame in starts:
            a = self.arrays[seed]
            if not 0 <= i < a.shape[0] or frame < 0 or frame + self.horizon >= a.shape[1]:
                raise ValueError("selected window exceeds trajectory")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        seed, i, frame = self.starts[index]
        return torch.tensor(self.arrays[seed][i, frame : frame + self.horizon + 1], dtype=torch.float32)


class SelectedDetStats(SelectedWindows):
    def __init__(self, root, starts):
        super().__init__(root, starts, 1)

    def __getitem__(self, index):
        window = super().__getitem__(index)
        return {"_seed": index, "init": window[0], "mean": window[1, :1], "std": torch.zeros_like(window[1, :1])}


class ManifestData(IncnsData):
    """Training retains complete seeds 42–44; fit always uses validation only."""

    def __init__(self, *, manifest, partition="validation", rollout_steps=10, **kwargs):
        if manifest is None:
            raise ValueError("provide an absolute frozen data.manifest")
        if partition not in ("validation", "test"):
            raise ValueError("partition must be validation or test")
        self.manifest, _ = load_manifest(manifest, kwargs["root"])
        self.partition = partition
        self.rollout_steps = int(rollout_steps)
        super().__init__(**kwargs)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and self.partition != "validation":
            raise ValueError("training and validation cannot use the final test partition")
        selection = self.manifest["evaluation"][self.partition]
        if self.rollout_steps not in selection["rollout_horizons"]:
            raise ValueError("rollout horizon is not frozen in this protocol")
        if stage in (None, "fit"):
            groups = self.manifest["partitions"]["train"]["trajectories"]
            for group in groups:
                a = np.load(self.root / f"traj_seed_{group['seed']}.npy", mmap_mode="r")
                if group["ids"] != list(range(a.shape[0])):
                    raise ValueError("this protocol preserves complete training files")
            self.train_set = TrajTrain(
                self.root,
                seeds=[g["seed"] for g in groups],
                stride=self.train_dataset_cfg.get("stride", 1),
                seq_len=self.train_dataset_cfg.get("seq_len", 2),
            )
        variant = self.manifest["variant"]
        members = self.manifest["partitions"][self.partition]
        stats = (
            StocStats(self.root, members["cases"])
            if variant == "stoc"
            else SelectedDetStats(self.root, selection["stats_pairs"])
        )
        rollout = SelectedWindows(self.root, selection["rollout_starts"], self.rollout_steps if variant == "det" else 0)
        self.eval_sets = {"stats": stats, "rollout": rollout}
        self.test_loader_cfgs = {"stats": {"batch_size": 1}, "rollout": {"batch_size": self.batch_size}}
        spectrum_path = self.root / selection["reference_spectrum"]
        spectrum = np.load(spectrum_path, allow_pickle=False)
        if spectrum.ndim != 2 or spectrum.shape[0] < self.rollout_steps:
            raise ValueError("invalid held-out spectrum reference")
        self.reference_spectrum = torch.tensor(spectrum[: self.rollout_steps], dtype=torch.float32)
