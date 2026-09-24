from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from model import (  # noqa: E402
    build_model_for_checkpoint,
    extract_model_state_dict,
    state_dict_uses_tied_weights,
)


# ============================================================
# EXPECTED NOVA ARCHITECTURE
# ============================================================

VOCAB_SIZE = 50257
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
BLOCK_SIZE = 512


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINTS = [
    {
        "name": "NOVA FINAL",
        "path": PROJECT_ROOT
        / "out"
        / "final"
        / "best_final.pt",
        "expected_tied": True,
    },
    {
        "name": "NOVA V1",
        "path": PROJECT_ROOT
        / "out"
        / "archive"
        / "nova_v1"
        / "checkpoint.pt",
        "expected_tied": False,
    },
    {
        "name": "TECHNICAL V1.1",
        "path": PROJECT_ROOT
        / "out"
        / "archive"
        / "technical_v1_1"
        / "best_checkpoint.pt",
        "expected_tied": False,
    },
]


# ============================================================
# UTILS
# ============================================================

def human_size(
    num_bytes: int,
) -> str:
    value = float(num_bytes)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:
        if value < 1024.0:
            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024.0

    return (
        f"{value:.2f} PB"
    )


def sha256_file(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def format_value(
    value: Any,
) -> str:
    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value)


def print_checkpoint_metadata(
    checkpoint: Any,
):
    if not isinstance(
        checkpoint,
        dict,
    ):
        print(
            "Checkpoint является raw state_dict."
        )
        return

    print(
        "Top-level keys:"
    )

    for key in checkpoint.keys():
        print(
            f"  - {key}"
        )

    interesting_keys = [
        "iter_num",
        "best_val_loss",
        "val_loss",
        "step",
        "epoch",
    ]

    print()
    print(
        "Metadata:"
    )

    found_any = False

    for key in interesting_keys:
        if key in checkpoint:
            found_any = True

            print(
                f"  {key}: "
                f"{format_value(checkpoint[key])}"
            )

    if not found_any:
        print(
            "  (основные metadata "
            "поля отсутствуют)"
        )

    if (
        "config"
        in checkpoint
        and isinstance(
            checkpoint["config"],
            dict,
        )
    ):
        cfg = checkpoint[
            "config"
        ]

        print()
        print(
            "Checkpoint config:"
        )

        for key in [
            "vocab_size",
            "embed_dim",
            "num_heads",
            "num_layers",
            "block_size",
            "dropout",
            "batch_size",
            "learning_rate",
            "max_iters",
            "dtype",
            "out_dir",
            "variant",
        ]:
            if key in cfg:
                print(
                    f"  {key}: "
                    f"{cfg[key]}"
                )


def inspect_state_dict(
    state_dict,
):
    print()
    print(
        "State dict:"
    )

    print(
        f"  tensors/entries: "
        f"{len(state_dict):,}"
    )

    required = [
        "token_embedding.weight",
        "position_embedding.weight",
        "head.weight",
        "ln_f.weight",
        "ln_f.bias",
    ]

    missing = [
        key
        for key in required
        if key not in state_dict
    ]

    if missing:
        print(
            "  ERROR: отсутствуют "
            "обязательные ключи:"
        )

        for key in missing:
            print(
                f"    - {key}"
            )

        raise KeyError(
            "Checkpoint не похож "
            "на ожидаемую NOVA."
        )

    token_weight = state_dict[
        "token_embedding.weight"
    ]

    position_weight = state_dict[
        "position_embedding.weight"
    ]

    head_weight = state_dict[
        "head.weight"
    ]

    print(
        "  token_embedding.weight: "
        f"{tuple(token_weight.shape)}"
    )

    print(
        "  position_embedding.weight: "
        f"{tuple(position_weight.shape)}"
    )

    print(
        "  head.weight: "
        f"{tuple(head_weight.shape)}"
    )

    if tuple(
        token_weight.shape
    ) != (
        VOCAB_SIZE,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Неожиданная форма "
            "token_embedding.weight: "
            f"{tuple(token_weight.shape)}"
        )

    if tuple(
        head_weight.shape
    ) != (
        VOCAB_SIZE,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Неожиданная форма "
            "head.weight: "
            f"{tuple(head_weight.shape)}"
        )

    if tuple(
        position_weight.shape
    ) != (
        BLOCK_SIZE,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Неожиданная форма "
            "position_embedding.weight: "
            f"{tuple(position_weight.shape)}"
        )


def check_storage_relation(
    state_dict,
):
    token_weight = state_dict[
        "token_embedding.weight"
    ]

    head_weight = state_dict[
        "head.weight"
    ]

    same_values = bool(
        torch.equal(
            token_weight,
            head_weight,
        )
    )

    same_storage = False

    try:
        same_storage = (
            token_weight
            .untyped_storage()
            .data_ptr()
            ==
            head_weight
            .untyped_storage()
            .data_ptr()
        )

    except Exception:
        pass

    print()
    print(
        "Embedding / LM head:"
    )

    print(
        f"  same values: "
        f"{same_values}"
    )

    print(
        f"  same storage in loaded ckpt: "
        f"{same_storage}"
    )

    return (
        same_values,
        same_storage,
    )


# ============================================================
# FORWARD TESTS
# ============================================================

@torch.inference_mode()
def cpu_forward_test(
    model,
):
    model = model.to(
        "cpu"
    )

    model.eval()

    x = torch.tensor(
        [
            [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
            ]
        ],
        dtype=torch.long,
    )

    logits, _ = model(
        x
    )

    expected_shape = (
        1,
        8,
        VOCAB_SIZE,
    )

    if tuple(
        logits.shape
    ) != expected_shape:
        raise RuntimeError(
            "CPU forward вернул "
            f"{tuple(logits.shape)}, "
            f"ожидалось {expected_shape}"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "CPU forward содержит "
            "NaN/Inf."
        )

    print(
        "  CPU forward: OK"
    )

    print(
        f"    logits shape: "
        f"{tuple(logits.shape)}"
    )

    print(
        f"    logits mean: "
        f"{logits.float().mean().item():.6f}"
    )

    print(
        f"    logits std: "
        f"{logits.float().std().item():.6f}"
    )


@torch.inference_mode()
def gpu_forward_test(
    model,
):
    if not torch.cuda.is_available():
        print(
            "  GPU forward: SKIP "
            "(CUDA недоступна)"
        )

        return

    device = torch.device(
        "cuda"
    )

    model = model.to(
        device
    )

    model.eval()

    x = torch.tensor(
        [
            [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    if torch.cuda.is_bf16_supported():
        amp_dtype = (
            torch.bfloat16
        )

        dtype_name = (
            "bfloat16"
        )

    else:
        amp_dtype = (
            torch.float16
        )

        dtype_name = (
            "float16"
        )

    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    ):
        logits, _ = model(
            x
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "GPU forward содержит "
            "NaN/Inf."
        )

    print(
        "  GPU forward: OK"
    )

    print(
        f"    autocast: "
        f"{dtype_name}"
    )

    print(
        f"    logits dtype: "
        f"{logits.dtype}"
    )

    print(
        f"    VRAM allocated: "
        f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
    )


# ============================================================
# SINGLE CHECKPOINT
# ============================================================

def inspect_checkpoint(
    item: dict,
):
    name = item[
        "name"
    ]

    path = item[
        "path"
    ]

    expected_tied = item[
        "expected_tied"
    ]

    print()
    print(
        "=" * 78
    )

    print(
        name
    )

    print(
        "=" * 78
    )

    print(
        f"Path: {path}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint не найден: "
            f"{path}"
        )

    file_size = (
        path.stat().st_size
    )

    print(
        f"Size: "
        f"{human_size(file_size)} "
        f"({file_size:,} bytes)"
    )

    print()
    print(
        "Calculating SHA-256..."
    )

    started = time.time()

    file_hash = sha256_file(
        path
    )

    print(
        f"SHA-256: "
        f"{file_hash}"
    )

    print(
        f"Hash time: "
        f"{time.time() - started:.1f}s"
    )

    print()
    print(
        "Loading checkpoint to CPU..."
    )

    started = time.time()

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        load_mode = (
            "weights_only=True"
        )

    except Exception as exc:
        print(
            "weights_only=True "
            "не сработал:"
        )

        print(
            f"  {type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "Повторная загрузка "
            "weights_only=False..."
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        load_mode = (
            "weights_only=False"
        )

    print(
        f"Loaded in "
        f"{time.time() - started:.1f}s "
        f"({load_mode})"
    )

    print_checkpoint_metadata(
        checkpoint
    )

    state_dict = (
        extract_model_state_dict(
            checkpoint
        )
    )

    inspect_state_dict(
        state_dict
    )

    (
        same_values,
        same_storage,
    ) = check_storage_relation(
        state_dict
    )

    detected_tied = (
        state_dict_uses_tied_weights(
            state_dict
        )
    )

    print(
        f"  detected architecture: "
        f"{'TIED' if detected_tied else 'UNTIED'}"
    )

    print(
        f"  expected architecture: "
        f"{'TIED' if expected_tied else 'UNTIED'}"
    )

    if (
        detected_tied
        != expected_tied
    ):
        raise RuntimeError(
            f"{name}: "
            "тип weight tying "
            "не совпал с ожидаемым."
        )

    print()
    print(
        "Building matching GPT..."
    )

    model, tie_weights = (
        build_model_for_checkpoint(
            checkpoint=checkpoint,
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            block_size=BLOCK_SIZE,
            dropout=0.0,
        )
    )

    print(
        f"  strict load: OK"
    )

    print(
        f"  tie_weights: "
        f"{tie_weights}"
    )

    print(
        f"  model.weights_are_tied(): "
        f"{model.weights_are_tied()}"
    )

    if (
        model.weights_are_tied()
        != expected_tied
    ):
        raise RuntimeError(
            "После построения модели "
            "weight tying неверен."
        )

    total_params = (
        model.num_params()
    )

    print(
        f"  unique parameters: "
        f"{total_params:,}"
    )

    print(
        f"  unique parameters: "
        f"{total_params / 1_000_000:.3f}M"
    )

    # Примерные sanity ranges.
    #
    # Здесь не делаем жёсткую проверку
    # на одно конкретное число,
    # но ловим очевидно неправильную
    # архитектуру.
    if expected_tied:
        if not (
            120_000_000
            <= total_params
            <= 130_000_000
        ):
            raise RuntimeError(
                "Для tied NOVA "
                "неожиданное число "
                f"параметров: "
                f"{total_params:,}"
            )

    else:
        if not (
            158_000_000
            <= total_params
            <= 166_000_000
        ):
            raise RuntimeError(
                "Для untied NOVA "
                "неожиданное число "
                f"параметров: "
                f"{total_params:,}"
            )

    print()
    print(
        "Forward tests:"
    )

    cpu_forward_test(
        model
    )

    gpu_forward_test(
        model
    )

    result = {
        "name":
            name,

        "path":
            str(path),

        "size":
            file_size,

        "sha256":
            file_hash,

        "detected_tied":
            detected_tied,

        "same_values":
            same_values,

        "same_storage":
            same_storage,

        "parameters":
            total_params,
    }

    del model
    del state_dict
    del checkpoint

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print(
        "=" * 78
    )

    print(
        "NOVA CHECKPOINT INTEGRITY TEST"
    )

    print(
        "=" * 78
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"BF16 supported: "
            f"{torch.cuda.is_bf16_supported()}"
        )

    tokenizer = (
        tiktoken.get_encoding(
            "gpt2"
        )
    )

    print(
        f"GPT-2 vocab: "
        f"{tokenizer.n_vocab}"
    )

    if (
        tokenizer.n_vocab
        != VOCAB_SIZE
    ):
        raise RuntimeError(
            "Неожиданный GPT-2 "
            "vocab size."
        )

    results = []

    failures = []

    for item in CHECKPOINTS:
        try:
            result = inspect_checkpoint(
                item
            )

            results.append(
                result
            )

        except Exception as exc:
            failures.append(
                (
                    item["name"],
                    type(exc).__name__,
                    str(exc),
                )
            )

            print()
            print(
                "!!! CHECK FAILED !!!"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print(
        "=" * 78
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    for result in results:
        print(
            f"{result['name']:<20} | "
            f"{'TIED' if result['detected_tied'] else 'UNTIED':<6} | "
            f"{result['parameters'] / 1_000_000:8.3f}M | "
            f"{human_size(result['size']):>10}"
        )

        print(
            f"  SHA-256: "
            f"{result['sha256']}"
        )

    if failures:
        print()
        print(
            "FAILURES:"
        )

        for (
            name,
            error_type,
            message,
        ) in failures:
            print(
                f"  {name}: "
                f"{error_type}: "
                f"{message}"
            )

        print()
        print(
            "=" * 78
        )

        print(
            "RESULT: FAILED"
        )

        print(
            "=" * 78
        )

        raise SystemExit(1)

    if (
        len(results)
        != len(CHECKPOINTS)
    ):
        raise RuntimeError(
            "Проверены не все "
            "checkpoint."
        )

    print()
    print(
        "=" * 78
    )

    print(
        "RESULT: ALL CHECKPOINTS OK"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Ни один checkpoint "
        "не был изменён."
    )

    print(
        "Ни один новый файл "
        "не был создан."
    )


if __name__ == "__main__":
    main()