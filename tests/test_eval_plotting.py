"""Plot held-out artifacts correctly under job-ID output directories."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import yaml

from stocbench.utils import plotting


class EvalPlottingTests(unittest.TestCase):
    def test_job_directory_uses_saved_model_and_actual_nfe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plots = root / "plots.yaml"
            plots.write_text("errors:\n  metrics: [{key: mean_error, ylabel: Error}]\n")
            for model, label, expected in (("dpm2", "DPM-2", 19), ("ddim", "DDIM", 10), ("ddpm", "DDPM", 10)):
                with self.subTest(model=model):
                    run = root / model / "123456"
                    artifacts = run / "artifacts"
                    artifacts.mkdir(parents=True)
                    (run / "config.yaml").write_text(yaml.safe_dump({"model": model}))
                    np.savez(artifacts / "mean_error_u_10.npz", mean_value=0.1)
                    with patch.object(plotting, "_apply_style"), patch.object(plotting, "plot_metrics_steps") as draw:
                        draw.return_value = None
                        plotting.plot_runs([run], plots_config=plots)
                    selected = draw.call_args.args[0][0]
                    self.assertEqual(selected.label, label)
                    steps, values = plotting._series(selected, "mean_error", lambda d: float(d["mean_value"]))
                    self.assertEqual(steps, (expected,))
                    self.assertEqual(values, (0.1,))

    def test_missing_legacy_cases_skip_panel_but_actual_cases_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts").mkdir()
            for stat in ("mean", "std"):
                np.savez(
                    root / f"artifacts/{stat}_error_u_10.npz",
                    **{
                        f"{stat}_prediction": np.ones((2, 1, 4, 4)),
                        f"{stat}_reference": np.zeros((2, 1, 4, 4)),
                        f"{stat}_seeds": np.array([20000, 20001]),
                    },
                )
            run = plotting.Run(root, "FM", None)
            self.assertIsNone(plotting.plot_seed_fields([run], {"seeds": [110, 111, 112]}))
            fig = plotting.plot_seed_fields([run], {"seeds": 2})
            self.assertIsNotNone(fig)
            self.assertTrue(any("20000" in ax.get_title() for ax in fig.axes))
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
