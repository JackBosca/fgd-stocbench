"""Small CPU checks: no released data, GPU access, training or Slurm submission."""

import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

from stocbench.protocol import (
    REPO,
    check_cross_partition,
    digest,
    freeze_release,
    item_path,
    read_json,
    sha256,
    source_evidence,
    spectrum_numpy,
    stream_key,
    validate_config,
    verify_frozen,
    write_json,
)


def tiny_config():
    c = read_json(REPO / "protocols/cfgd-stocbench-v1.json")
    c.update(grid=4, warmup_frames=2, trajectory_frames=60, trajectories_per_partition=2, reference_members=3)
    c["solver"].update(n=8, dt=0.001, sample_dt=0.002)
    c["partitions"]["validation"]["cases"] = 1
    c["partitions"]["test"]["cases"] = 2
    return c


def fixture(release):
    from solver.generate_holdouts import save_record

    c = tiny_config()
    plan = {"schema_version": 1, "config": c, "training": {}, "generator": source_evidence()}
    rng = np.random.default_rng(9317)
    for variant in ("stoc", "det"):
        channels = 1 if variant == "stoc" else 2
        plan["training"][variant] = []
        for seed in (42, 43, 44):
            path = release / f"old-{variant}-{seed}.npy"
            a = rng.normal(size=(2, 60, channels, 4, 4)).astype(np.float32)
            np.save(path, a)
            plan["training"][variant].append({"seed": seed, "path": str(path), "shape": list(a.shape)})
    plan["plan_sha256"] = digest(plan)
    write_json(release / "plan.json", plan)
    for variant in ("stoc", "det"):
        channels = 1 if variant == "stoc" else 2
        for partition in ("validation", "test"):
            for i in range(2):
                path = item_path(release, variant, partition, "trajectory", i)
                path.mkdir(parents=True)
                np.save(path / "trajectory.npy", rng.normal(size=(60, channels, 4, 4)).astype(np.float32))
                save_record(path, plan, variant, partition, "trajectory", i, {"fixture": True})
            if variant == "stoc":
                p = c["partitions"][partition]
                for case in range(p["case_start"], p["case_start"] + p["cases"]):
                    path = item_path(release, variant, partition, "case", case)
                    path.mkdir(parents=True)
                    high = rng.normal(size=(1, 8, 8)).astype(np.float32)
                    np.save(path / "init_highres.npy", high)
                    np.save(path / "init.npy", high.reshape(1, 4, 2, 4, 2).mean((2, 4)))
                    np.save(path / "raw.npy", rng.normal(size=(3, 1, 1, 4, 4)).astype(np.float32))
                    save_record(path, plan, variant, partition, "case", case, {"fixture": True})
    return plan


class ProtocolTests(unittest.TestCase):
    def test_streams_separate_and_sharding_preserves_identity(self):
        c = tiny_config()
        keys = [
            tuple(stream_key(c, v, p, "case", i, phase, j))
            for v in ("stoc", "det")
            for p in ("validation", "test")
            for i in range(3)
            for phase in ("warmup", "future")
            for j in range(4)
        ]
        self.assertEqual(len(keys), len(set(keys)))
        serial = {i: stream_key(c, "stoc", "test", "trajectory", i, "warmup").tolist() for i in range(10)}
        sharded = {
            i: stream_key(c, "stoc", "test", "trajectory", i, "warmup").tolist()
            for shard in range(3)
            for i in range(10)[shard::3]
        }
        self.assertEqual(serial, sharded)

    def test_rejects_overlap_and_bad_horizon(self):
        c = tiny_config()
        validate_config(c)
        c["partitions"]["test"]["trajectory_seed"] = 10000
        with self.assertRaises(ValueError):
            validate_config(c)
        c = tiny_config()
        c["rollout_start"] = 20
        with self.assertRaises(ValueError):
            validate_config(c)
        seen = {}
        check_cross_partition(np.zeros((1, 4, 4), np.float32), "validation", seen)
        with self.assertRaises(ValueError):
            check_cross_partition(np.zeros((1, 4, 4), np.float32), "test", seen)

    def test_freeze_loaders_spectrum_and_checkpoint_lock(self):
        import torch

        from stocbench.data.heldout import ManifestData, SelectedDetStats, load_manifest
        from stocbench.heldout import evaluation_command, lock_baselines
        from stocbench.metrics import radial_enstrophy

        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            plan = fixture(release)
            freeze_release(release)
            verify_frozen(release / "frozen")
            m, root = load_manifest(release / "frozen/det.json")
            self.assertEqual(m["partitions"]["test"]["trajectories"][0]["ids"], [0, 1])
            stats = SelectedDetStats(root, m["evaluation"]["test"]["stats_pairs"])
            a = np.load(root / "traj_seed_10001.npy", mmap_mode="r")
            np.testing.assert_array_equal(stats[1]["init"].numpy(), a[1, 0])
            np.testing.assert_array_equal(stats[1]["mean"].numpy(), a[1, 1, :1])
            spec = spectrum_numpy(a[:, 1, 0])
            np.testing.assert_allclose(spec, radial_enstrophy(torch.tensor(a[:, 1, 0])).numpy(), rtol=1e-5, atol=1e-6)
            expected = sum(spectrum_numpy(x[1:51, 0]) for x in a) / len(a)
            np.testing.assert_allclose(np.load(root / "enstrophy_test.npy"), expected, rtol=1e-6)
            data = ManifestData(
                root=str(root),
                manifest=str(release / "frozen/det.json"),
                partition="test",
                rollout_steps=10,
                sim_channels=["vorticity"],
                sim_params=[],
                train_dataset={},
            )
            with self.assertRaises(ValueError):
                data.setup("fit")
            data.setup("test")
            self.assertEqual(len(data.eval_sets["rollout"]), 2)
            self.assertEqual(data.eval_sets["rollout"][1].shape, (11, 2, 4, 4))
            weight = release / "weights.ckpt"
            weight.write_bytes(b"test weights, never loaded")
            spec_path, lock_path = release / "weights.json", release / "lock.json"
            write_json(
                spec_path,
                {
                    "variant": "det",
                    "baselines": ["fm"],
                    "checkpoints": {"fm": {"path": str(weight), "provenance": "unit fixture"}},
                },
            )
            lock_baselines(spec_path, lock_path)
            command, evidence = evaluation_command(
                release / "frozen/det.json", lock_path, "fm", "test", release / "run"
            )
            self.assertIn("data.partition=test", command)
            self.assertIn("evaluate.n_inits=2", command)
            self.assertIn("evaluate.ensemble_size=3000", command)
            self.assertEqual(evidence["partition"], "test")
            weight.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                evaluation_command(release / "frozen/det.json", lock_path, "fm", "test", release / "run")
            # Original training arrays are linked and hash-checked, never copied/rewritten.
            self.assertTrue((root / "traj_seed_42.npy").is_symlink())
            with self.assertRaises(FileExistsError):
                freeze_release(release)
            changed = Path(plan["training"]["det"][0]["path"])
            with changed.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaises(ValueError):
                verify_frozen(release / "frozen")

    def test_real_callback_population_aggregation(self):
        import pytorch_lightning as pl

        from stocbench.callbacks.eval import EvalCallback
        from stocbench.data.heldout import ManifestData, verify_selection
        from stocbench.metrics import EnsembleStatError, EnstrophyError, RolloutRMSE

        class Persistence(pl.LightningModule):
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                return None

            def sample(self, batch, sampling_schedule):
                return batch[:, :1]

        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            fixture(release)
            freeze_release(release)
            root = release / "frozen/incns_det/64"
            manifest = release / "frozen/det.json"
            verify_selection(manifest, "test")
            data = ManifestData(
                root=str(root),
                manifest=str(manifest),
                partition="test",
                rollout_steps=10,
                sim_channels=["vorticity"],
                sim_params=[],
                train_dataset={},
                batch_size=2,
                num_workers=0,
            )
            output = release / "eval"
            output.mkdir()
            callback = EvalCallback(
                [EnsembleStatError("mean"), EnstrophyError(), RolloutRMSE()],
                ensemble_size=3,
                rollout_steps=10,
                batch_size=2,
                gt_rollout=True,
                output_dir=output,
            )
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[callback],
            )
            trainer.test(Persistence(), datamodule=data)
            with np.load(output / "artifacts/enstrophy_error_u_1.npz") as result:
                a = np.load(root / "traj_seed_10001.npy", mmap_mode="r")
                expected = spectrum_numpy(a[:, 0, 0]).mean(0)
                np.testing.assert_allclose(
                    result["enstr_model_spectrum"], np.repeat(expected[None], 10, axis=0), rtol=1e-5, atol=1e-6
                )
            with np.load(output / "artifacts/mean_error_u_1.npz") as result:
                self.assertEqual(result["mean_per_initial_condition"].shape, (2,))
            with (root / "enstrophy_test.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                verify_selection(manifest, "test")

    def test_corrupt_shard_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp)
            fixture(release)
            path = item_path(release, "stoc", "validation", "trajectory", 0) / "trajectory.npy"
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                freeze_release(release)
            self.assertFalse((release / "frozen").exists())
            self.assertFalse((release / ".freeze.lock").exists())

    def test_slurm_preview_cannot_submit_or_compute(self):
        for script, args in (
            ("generate.sh", ["--release", "/tmp/unused", "--variant", "stoc", "--partition", "test", "--kind", "case"]),
            ("verify.sh", ["freeze", "--release", "/tmp/unused"]),
            ("evaluate.sh", ["--help"]),
        ):
            env = dict(os.environ, STOCBENCH_DRY_RUN="1", STOCBENCH_REPO=str(REPO))
            result = subprocess.run(
                ["bash", str(REPO / "slurm/heldout" / script), *args],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Launch: srun", result.stdout)
            self.assertNotIn("sbatch", result.stdout)


class EngineTests(unittest.TestCase):
    def test_batching_preserves_streams_and_det_forcing_alignment(self):
        try:
            from solver.heldout_engine import HeldoutEngine
        except ModuleNotFoundError as error:
            self.skipTest(f"optional JAX dependency unavailable: {error}")
        c = tiny_config()
        c["trajectory_frames"] = 3
        for variant in ("stoc", "det"):
            engine = HeldoutEngine(c, variant)
            warm = np.stack([stream_key(c, variant, "test", "trajectory", i, "warmup") for i in range(2)])
            keys = np.stack([stream_key(c, variant, "test", "trajectory", i, "trajectory") for i in range(2)])
            together = engine.trajectories(warm, keys)
            separately = np.concatenate([engine.trajectories(warm[i : i + 1], keys[i : i + 1]) for i in range(2)])
            np.testing.assert_allclose(together, separately, rtol=2e-5, atol=2e-6)
            self.assertTrue(np.isfinite(together).all())
            if variant == "det":
                # At full resolution, advance stored omega with its stored forcing.
                import jax.numpy as jnp

                full_config = copy.deepcopy(c)
                full_config["grid"] = 8
                full = HeldoutEngine(full_config, "det")
                a = full.trajectories(warm[:1], keys[:1])
                wh = full.fft(jnp.asarray(a[:, 0, 0] * c["solver"]["std"]))
                forcing = full.fft(jnp.asarray(a[:, 0, 1]))
                prediction = full.normalized(full.deterministic(wh, forcing))
                np.testing.assert_allclose(prediction, a[:, 1, 0], rtol=2e-5, atol=2e-6)

    def test_compiled_stochastic_step_matches_upstream_update(self):
        try:
            import jax
            import jax.numpy as jnp
            from solver.heldout_engine import HeldoutEngine
        except ModuleNotFoundError as error:
            self.skipTest(f"optional JAX dependency unavailable: {error}")
        c = tiny_config()
        engine = HeldoutEngine(c, "stoc")
        wh = engine.fft(jnp.asarray(np.random.default_rng(23).normal(size=(1, 8, 8)).astype(np.float32)))
        key = jax.random.PRNGKey(123)
        expected, original_key = wh, key
        for _ in range(engine.solver.steps):
            expected, original_key = engine.solver._step_hat(expected, original_key)
        actual, keys = engine.stochastic(wh, key[None])
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(keys[0]), np.asarray(original_key))

    def test_case_future_batching_and_resume(self):
        try:
            from solver.generate_holdouts import generate
            from solver.heldout_engine import HeldoutEngine
        except ModuleNotFoundError as error:
            self.skipTest(f"optional JAX dependency unavailable: {error}")
        c = tiny_config()
        e = HeldoutEngine(c, "stoc")
        high = e.condition(
            stream_key(c, "stoc", "test", "case", 21000, "warmup"),
            stream_key(c, "stoc", "test", "case", 21000, "condition"),
        )
        keys = np.stack([stream_key(c, "stoc", "test", "case", 21000, "future", i) for i in range(3)])
        a = e.futures(high, keys)
        b = np.concatenate([e.futures(high, keys[i : i + 1]) for i in range(3)])
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
        self.assertFalse(np.array_equal(a[0], a[1]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"schema_version": 1, "config": c, "training": {}, "generator": source_evidence()}
            plan["plan_sha256"] = digest(plan)
            write_json(root / "plan.json", plan)
            generate(root, "stoc", "validation", "case", member_batch=2)
            path = item_path(root, "stoc", "validation", "case", 20000) / "raw.npy"
            original = sha256(path)
            with patch("solver.heldout_engine.HeldoutEngine", side_effect=AssertionError("must not recompute")):
                generate(root, "stoc", "validation", "case", member_batch=1)
            self.assertEqual(original, sha256(path))


if __name__ == "__main__":
    unittest.main()
