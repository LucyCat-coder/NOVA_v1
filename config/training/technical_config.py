from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TECHNICAL_SFT_CONFIG = {
    # ========================================================
    # SOURCE MODEL
    # ========================================================

    # Исходный FINAL checkpoint никогда
    # не перезаписывается.
    "base_checkpoint": (
        PROJECT_ROOT
        / "out"
        / "final"
        / "best_final.pt"
    ),

    # ========================================================
    # DATA
    # ========================================================

    "train_data_path": (
        PROJECT_ROOT
        / "data"
        / "nova_technical_v1_1"
        / "splits"
        / "train.jsonl"
    ),

    "val_data_path": (
        PROJECT_ROOT
        / "data"
        / "nova_technical_v1_1"
        / "splits"
        / "val.jsonl"
    ),

    "tokenizer": "gpt2",

    "eos_token_id": 50256,

    "ignore_index": -100,

    "block_size": 512,

    # ========================================================
    # MODEL / TRAINING
    # ========================================================

    "device": "cuda",
    "dtype": "bfloat16",

    # Dropout совпадает с основной
    # конфигурацией обучения FINAL.
    "dropout": 0.1,

    "batch_size": 4,

    # Effective batch:
    # 4 * 8 = 32 examples.
    "gradient_accumulation_steps": 8,

    # Консервативный LR для SFT
    # поверх уже обученного checkpoint.
    "learning_rate": 1e-5,

    "weight_decay": 0.1,

    "betas": (
        0.9,
        0.95,
    ),

    "grad_clip": 1.0,

    # Это пока pilot-конфигурация.
    #
    # НЕ запускаем её до проверки dataset.
    "max_steps": 120,

    "eval_interval": 20,

    "log_interval": 5,

    "seed": 1337,

    "num_workers": 0,

    # ========================================================
    # OUTPUT
    # ========================================================

    "out_dir": (
        PROJECT_ROOT
        / "out"
        / "candidate_sft"
        / "technical_v1"
    ),

    "report_path": (
        PROJECT_ROOT
        / "reports"
        / "technical_sft_training.txt"
    ),
}