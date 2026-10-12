"""Plain-text metric summaries for non-interactive training logs."""

from datetime import UTC, datetime

import pytorch_lightning as pl


class TextMetrics(pl.Callback):
    """Print epoch/validation metrics on rank zero, with optional batch updates.

    Values retain the model's existing aggregation semantics (for example,
    train_loss_epoch is rank zero's average, not a new distributed reduction).
    """

    def __init__(self, every_n_train_batches: int = 0) -> None:
        if every_n_train_batches < 0:
            raise ValueError("every_n_train_batches must be nonnegative")
        self.every_n_train_batches = every_n_train_batches

    @staticmethod
    def _print(trainer, phase: str, select, *, batch: int | None = None) -> None:
        # Access on every rank: Lightning may synchronize logged metric values.
        metrics = trainer.callback_metrics
        if not trainer.is_global_zero:
            return
        values = " ".join(f"{key}={float(value):.6g}" for key, value in sorted(metrics.items()) if select(key))
        if not values:
            return
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        position = f"epoch={trainer.current_epoch} step={trainer.global_step}"
        if batch is not None:
            position += f" batch={batch}"
        print(f"{timestamp} [{phase}] {position} {values}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.every_n_train_batches and (batch_idx + 1) % self.every_n_train_batches == 0:
            self._print(
                trainer,
                "train batch",
                lambda key: key.endswith("_step") and key.startswith(("train_loss", "train/")),
                batch=batch_idx + 1,
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._print(
            trainer,
            "train epoch",
            lambda key: key.endswith("_epoch") and key.startswith(("train_loss", "train/")),
        )

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._print(trainer, "validation", lambda key: key.startswith("val/"))
