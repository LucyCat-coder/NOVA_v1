from __future__ import annotations

import argparse
import gc
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import MODEL_CONFIG
from config.training.technical_config import (
    TECHNICAL_SFT_CONFIG,
)
from model import (
    DEFAULT_IGNORE_INDEX,
    build_model_for_checkpoint,
)
from train.technical_dataset import (
    TechnicalSFTDataset,
)


def load_checkpoint(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception as exc:
        print(
            "[WARN] weights_only=True "
            "не сработал:"
        )

        print(
            f"       "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "[WARN] Повторная загрузка "
            "weights_only=False."
        )

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def autocast_context(
    device: torch.device,
    dtype_name: str,
):
    if device.type != "cuda":
        return nullcontext()

    if dtype_name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Запрошен bfloat16, "
                "но GPU его не поддерживает."
            )

        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if dtype_name == "float16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if dtype_name == "float32":
        return nullcontext()

    raise ValueError(
        f"Неизвестный dtype: "
        f"{dtype_name!r}"
    )


@torch.inference_mode()
def evaluate(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    dtype_name: str,
) -> float:
    was_training = (
        model.training
    )

    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in loader:
        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        with autocast_context(
            device,
            dtype_name,
        ):
            _, loss = model(
                input_ids,
                labels,
            )

        if loss is None:
            raise RuntimeError(
                "Model не вернул loss."
            )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Validation loss = NaN/Inf."
            )

        total_loss += float(
            loss.item()
        )

        batches += 1

    if was_training:
        model.train()

    if batches == 0:
        raise RuntimeError(
            "Validation DataLoader пуст."
        )

    return (
        total_loss
        / batches
    )


def serialize_config(
    config: dict,
) -> dict:
    result = {}

    for key, value in (
        config.items()
    ):
        if isinstance(
            value,
            Path,
        ):
            result[key] = str(
                value
            )

        elif isinstance(
            value,
            tuple,
        ):
            result[key] = list(
                value
            )

        else:
            result[key] = value

    return result


def save_candidate(
    *,
    path: Path,
    model,
    base_checkpoint_path: Path,
    base_iter_num: int | None,
    sft_step: int,
    val_loss: float,
    best_val_loss: float,
    config: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_iter = (
        (
            int(base_iter_num)
            + int(sft_step)
        )
        if base_iter_num
        is not None
        else int(sft_step)
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        # Compatibility field.
        "iter_num":
            combined_iter,

        "best_val_loss":
            float(
                best_val_loss
            ),

        # Explicit SFT metadata.
        "stage":
            "assistant_sft",

        "base_checkpoint":
            str(
                base_checkpoint_path
            ),

        "base_iter_num":
            base_iter_num,

        "sft_step":
            int(
                sft_step
            ),

        "sft_val_loss":
            float(
                val_loss
            ),

        "sft_config":
            serialize_config(
                config
            ),
    }

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temp_path,
    )

    temp_path.replace(
        path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Override max_steps "
            "из technical_config.py"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить перезапись "
            "существующих candidate checkpoints."
        ),
    )

    args = parser.parse_args()

    config = dict(
        TECHNICAL_SFT_CONFIG
    )

    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError(
                "--steps должен быть > 0."
            )

        config[
            "max_steps"
        ] = args.steps

    # ========================================================
    # PATHS
    # ========================================================

    base_checkpoint_path = Path(
        config[
            "base_checkpoint"
        ]
    )

    train_path = Path(
        config[
            "train_data_path"
        ]
    )

    val_path = Path(
        config[
            "val_data_path"
        ]
    )

    out_dir = Path(
        config[
            "out_dir"
        ]
    )

    report_path = Path(
        config[
            "report_path"
        ]
    )

    best_path = (
        out_dir
        / "best_candidate.pt"
    )

    last_path = (
        out_dir
        / "last_candidate.pt"
    )

    if not base_checkpoint_path.exists():
        raise FileNotFoundError(
            "Base checkpoint не найден: "
            f"{base_checkpoint_path}"
        )

    if not train_path.exists():
        raise FileNotFoundError(
            "Train dataset не найден: "
            f"{train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            "Validation dataset не найден: "
            f"{val_path}"
        )

    if not args.force:
        existing = [
            path
            for path in (
                best_path,
                last_path,
            )
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Candidate checkpoint уже "
                "существует. "
                "Ничего не перезаписано:\n"
                + "\n".join(
                    str(path)
                    for path in existing
                )
                + "\nЕсли перезапись действительно "
                "нужна, используй --force."
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SEEDS
    # ========================================================

    seed = int(
        config[
            "seed"
        ]
    )

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.set_float32_matmul_precision(
        "high"
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        config[
            "device"
        ]
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA указана в config, "
            "но недоступна."
        )

    # ========================================================
    # DATA
    # ========================================================

    print()
    print("=" * 88)
    print(
        "NOVA — TECHNICAL ASSISTANT-ONLY SFT"
    )
    print("=" * 88)

    print(
        f"Base checkpoint: "
        f"{base_checkpoint_path}"
    )

    print(
        f"Train:           "
        f"{train_path}"
    )

    print(
        f"Validation:      "
        f"{val_path}"
    )

    print(
        f"Output:          "
        f"{out_dir}"
    )

    print()

    print(
        "Loading datasets..."
    )

    train_dataset = (
        TechnicalSFTDataset(
            train_path,
            block_size=(
                config[
                    "block_size"
                ]
            ),
            tokenizer_name=(
                config[
                    "tokenizer"
                ]
            ),
            eos_token_id=(
                config[
                    "eos_token_id"
                ]
            ),
            ignore_index=(
                config[
                    "ignore_index"
                ]
            ),
        )
    )

    val_dataset = (
        TechnicalSFTDataset(
            val_path,
            block_size=(
                config[
                    "block_size"
                ]
            ),
            tokenizer_name=(
                config[
                    "tokenizer"
                ]
            ),
            eos_token_id=(
                config[
                    "eos_token_id"
                ]
            ),
            ignore_index=(
                config[
                    "ignore_index"
                ]
            ),
        )
    )

    print(
        f"Train examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val examples:   "
        f"{len(val_dataset):,}"
    )

    batch_size = int(
        config[
            "batch_size"
        ]
    )

    num_workers = int(
        config[
            "num_workers"
        ]
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading base checkpoint..."
    )

    checkpoint = (
        load_checkpoint(
            base_checkpoint_path
        )
    )

    base_iter_num = (
        checkpoint.get(
            "iter_num",
            None,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else None
    )

    model, tie_weights = (
        build_model_for_checkpoint(
            checkpoint=checkpoint,
            vocab_size=(
                MODEL_CONFIG[
                    "vocab_size"
                ]
            ),
            embed_dim=(
                MODEL_CONFIG[
                    "embed_dim"
                ]
            ),
            num_heads=(
                MODEL_CONFIG[
                    "num_heads"
                ]
            ),
            num_layers=(
                MODEL_CONFIG[
                    "num_layers"
                ]
            ),
            block_size=(
                MODEL_CONFIG[
                    "block_size"
                ]
            ),
            dropout=float(
                config[
                    "dropout"
                ]
            ),
            ignore_index=int(
                config[
                    "ignore_index"
                ]
            ),
        )
    )

    del checkpoint

    gc.collect()

    if not tie_weights:
        raise RuntimeError(
            "Base checkpoint неожиданно "
            "UNTIED. "
            "Technical SFT должен стартовать "
            "из FINAL tied checkpoint."
        )

    if (
        model.ignore_index
        != DEFAULT_IGNORE_INDEX
    ):
        raise RuntimeError(
            "Unexpected model ignore_index: "
            f"{model.ignore_index}"
        )

    model.to(
        device
    )

    model.train()

    print(
        "Architecture:    TIED"
    )

    print(
        f"Parameters:      "
        f"{model.num_params() / 1_000_000:.3f}M"
    )

    print(
        f"Ignore index:    "
        f"{model.ignore_index}"
    )

    print(
        f"Device:          "
        f"{device}"
    )

    print(
        f"Dtype/autocast:  "
        f"{config['dtype']}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = (
        model.configure_optimizers(
            weight_decay=float(
                config[
                    "weight_decay"
                ]
            ),
            learning_rate=float(
                config[
                    "learning_rate"
                ]
            ),
            betas=tuple(
                config[
                    "betas"
                ]
            ),
            device=str(
                device
            ),
        )
    )

    gradient_accumulation_steps = int(
        config[
            "gradient_accumulation_steps"
        ]
    )

    max_steps = int(
        config[
            "max_steps"
        ]
    )

    eval_interval = int(
        config[
            "eval_interval"
        ]
    )

    log_interval = int(
        config[
            "log_interval"
        ]
    )

    grad_clip = float(
        config[
            "grad_clip"
        ]
    )

    effective_batch_size = (
        batch_size
        * gradient_accumulation_steps
    )

    print()
    print(
        f"Batch size:      "
        f"{batch_size}"
    )

    print(
        f"Grad accum:      "
        f"{gradient_accumulation_steps}"
    )

    print(
        f"Effective batch: "
        f"{effective_batch_size}"
    )

    print(
        f"Learning rate:   "
        f"{config['learning_rate']}"
    )

    print(
        f"Max steps:       "
        f"{max_steps}"
    )

    print()

    # ========================================================
    # BASELINE EVAL
    # ========================================================

    print(
        "Evaluating untouched FINAL "
        "checkpoint on SFT validation..."
    )

    baseline_val_loss = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        dtype_name=(
            config[
                "dtype"
            ]
        ),
    )

    best_val_loss = float(
        baseline_val_loss
    )

    print(
        f"Baseline val loss: "
        f"{baseline_val_loss:.6f}"
    )

    print()

    report_lines = [
        "=" * 88,
        "NOVA TECHNICAL ASSISTANT-ONLY SFT",
        "=" * 88,
        "",
        f"Base checkpoint: "
        f"{base_checkpoint_path}",
        f"Base iter_num: "
        f"{base_iter_num}",
        f"Architecture: TIED",
        f"Train examples: "
        f"{len(train_dataset)}",
        f"Val examples: "
        f"{len(val_dataset)}",
        f"Batch size: "
        f"{batch_size}",
        f"Gradient accumulation: "
        f"{gradient_accumulation_steps}",
        f"Effective batch: "
        f"{effective_batch_size}",
        f"Learning rate: "
        f"{config['learning_rate']}",
        f"Max steps: "
        f"{max_steps}",
        f"Ignore index: "
        f"{model.ignore_index}",
        f"Baseline val loss: "
        f"{baseline_val_loss:.6f}",
        "",
    ]

    # ========================================================
    # TRAINING
    # ========================================================

    train_iterator = iter(
        train_loader
    )

    total_started = (
        time.time()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    for step in range(
        1,
        max_steps + 1,
    ):
        step_started = (
            time.time()
        )

        micro_losses = []

        for _ in range(
            gradient_accumulation_steps
        ):
            try:
                batch = next(
                    train_iterator
                )

            except StopIteration:
                train_iterator = iter(
                    train_loader
                )

                batch = next(
                    train_iterator
                )

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=True,
            )

            with autocast_context(
                device,
                config[
                    "dtype"
                ],
            ):
                _, loss = model(
                    input_ids,
                    labels,
                )

            if loss is None:
                raise RuntimeError(
                    "Model не вернул loss."
                )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Training loss = NaN/Inf "
                    f"на step {step}."
                )

            raw_loss = float(
                loss.detach().item()
            )

            micro_losses.append(
                raw_loss
            )

            scaled_loss = (
                loss
                / gradient_accumulation_steps
            )

            scaled_loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        mean_train_loss = (
            sum(
                micro_losses
            )
            / len(
                micro_losses
            )
        )

        step_time = (
            time.time()
            - step_started
        )

        if (
            step == 1
            or step % log_interval == 0
        ):
            print(
                f"step "
                f"{step:4d}/"
                f"{max_steps} | "
                f"loss "
                f"{mean_train_loss:.6f} | "
                f"grad "
                f"{float(grad_norm):.4f} | "
                f"{step_time:.2f}s"
            )

            report_lines.append(
                f"step={step} "
                f"train_loss="
                f"{mean_train_loss:.6f} "
                f"grad_norm="
                f"{float(grad_norm):.6f} "
                f"step_time="
                f"{step_time:.3f}s"
            )

        should_eval = (
            step % eval_interval == 0
            or step == max_steps
        )

        if should_eval:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                dtype_name=(
                    config[
                        "dtype"
                    ]
                ),
            )

            improved = (
                val_loss
                < best_val_loss
            )

            print(
                f"  validation: "
                f"{val_loss:.6f}"
                + (
                    "  <-- BEST"
                    if improved
                    else ""
                )
            )

            report_lines.append(
                f"step={step} "
                f"val_loss="
                f"{val_loss:.6f} "
                f"improved="
                f"{improved}"
            )

            if improved:
                best_val_loss = float(
                    val_loss
                )

                save_candidate(
                    path=best_path,
                    model=model,
                    base_checkpoint_path=(
                        base_checkpoint_path
                    ),
                    base_iter_num=(
                        base_iter_num
                    ),
                    sft_step=step,
                    val_loss=val_loss,
                    best_val_loss=(
                        best_val_loss
                    ),
                    config=config,
                )

                print(
                    f"  saved: "
                    f"{best_path}"
                )

            model.train()

    # ========================================================
    # FINAL SAVE
    # ========================================================

    final_val_loss = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        dtype_name=(
            config[
                "dtype"
            ]
        ),
    )

    save_candidate(
        path=last_path,
        model=model,
        base_checkpoint_path=(
            base_checkpoint_path
        ),
        base_iter_num=(
            base_iter_num
        ),
        sft_step=max_steps,
        val_loss=final_val_loss,
        best_val_loss=(
            best_val_loss
        ),
        config=config,
    )

    total_time = (
        time.time()
        - total_started
    )

    report_lines.extend(
        [
            "",
            "=" * 88,
            "SUMMARY",
            "=" * 88,
            f"Baseline val loss: "
            f"{baseline_val_loss:.6f}",
            f"Best val loss: "
            f"{best_val_loss:.6f}",
            f"Final val loss: "
            f"{final_val_loss:.6f}",
            f"Total time: "
            f"{total_time:.2f}s",
            f"Best checkpoint: "
            f"{best_path}",
            f"Last checkpoint: "
            f"{last_path}",
            "",
        ]
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "TECHNICAL SFT FINISHED"
    )
    print("=" * 88)

    print(
        f"Baseline val loss: "
        f"{baseline_val_loss:.6f}"
    )

    print(
        f"Best val loss:     "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Final val loss:    "
        f"{final_val_loss:.6f}"
    )

    print(
        f"Total time:        "
        f"{total_time:.2f}s"
    )

    print(
        f"Report:            "
        f"{report_path}"
    )

    print(
        f"Best:              "
        f"{best_path}"
    )

    print(
        f"Last:              "
        f"{last_path}"
    )


if __name__ == "__main__":
    main()