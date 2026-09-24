from __future__ import annotations

import gc
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from config import MODEL_CONFIG  # noqa: E402
from model import (  # noqa: E402
    DEFAULT_IGNORE_INDEX,
    build_model_for_checkpoint,
)


# ============================================================================
# PATHS
# ============================================================================

FINAL_CHECKPOINT = (
    PROJECT_ROOT
    / "out"
    / "final"
    / "best_final.pt"
)

SFT_CHECKPOINT = (
    PROJECT_ROOT
    / "out"
    / "candidate_sft"
    / "technical_v1_step120"
    / "best_candidate.pt"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "nova_technical_v1_1"
    / "splits"
    / "train.jsonl"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "nova_technical_v1_1"
    / "splits"
    / "val.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sft_teacher_forcing_diagnostic.txt"
)


# ============================================================================
# CONSTANTS
# ============================================================================

EOS_TOKEN_ID = 50256
IGNORE_INDEX = DEFAULT_IGNORE_INDEX

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

TOKENIZER = tiktoken.get_encoding("gpt2")

BLOCK_SIZE = int(
    MODEL_CONFIG["block_size"]
)

ROLLOUT_MAX_TOKENS = 120

# Равномерно распределённые реальные TRAIN-примеры.
ROLLOUT_SAMPLE_COUNT = 8


# ============================================================================
# DATA
# ============================================================================

def read_jsonl(
    path: Path,
) -> list[dict]:
    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    f"Record {line_number} "
                    f"is not an object."
                )

            records.append(
                record
            )

    return records


def extract_qa(
    record: dict,
) -> tuple[str, str]:
    """
    Поддерживает оба формата:

    1)
        {
            "question": "...",
            "answer": "..."
        }

    2)
        {
            "messages": [
                {"role": "user", ...},
                {"role": "assistant", ...}
            ]
        }
    """

    question = record.get(
        "question"
    )

    answer = record.get(
        "answer"
    )

    if (
        isinstance(question, str)
        and isinstance(answer, str)
    ):
        return (
            question.strip(),
            answer.strip(),
        )

    messages = record.get(
        "messages"
    )

    if not isinstance(
        messages,
        list,
    ):
        raise ValueError(
            "Record does not contain "
            "question/answer or messages."
        )

    user_text = None
    assistant_text = None

    for message in messages:
        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = message.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):
            continue

        if (
            role == "user"
            and user_text is None
        ):
            user_text = (
                content.strip()
            )

        elif (
            role == "assistant"
            and assistant_text is None
        ):
            assistant_text = (
                content.strip()
            )

    if (
        not user_text
        or not assistant_text
    ):
        raise ValueError(
            "Could not extract "
            "user/assistant pair."
        )

    return (
        user_text,
        assistant_text,
    )


def encode_text(
    text: str,
) -> list[int]:
    return TOKENIZER.encode(
        text,
        disallowed_special=(),
    )


def decode_tokens(
    token_ids: list[int],
) -> str:
    raw = b"".join(
        TOKENIZER.decode_single_token_bytes(
            int(token_id)
        )
        for token_id in token_ids
    )

    return raw.decode(
        "utf-8",
        errors="replace",
    )


def build_example(
    record: dict,
) -> dict:
    question, answer = (
        extract_qa(
            record
        )
    )

    prompt = (
        f"Пользователь: {question}\n"
        f"Нова:"
    )

    prompt_ids = (
        encode_text(
            prompt
        )
    )

    # Именно так был построен SFT:
    # перед ответом добавляется пробел.
    answer_ids = (
        encode_text(
            " " + answer
        )
    )

    if len(prompt_ids) >= BLOCK_SIZE:
        raise ValueError(
            "Prompt itself is too long "
            f"for block size {BLOCK_SIZE}."
        )

    max_answer_tokens = (
        BLOCK_SIZE
        - len(prompt_ids)
    )

    truncated = (
        len(answer_ids)
        > max_answer_tokens
    )

    if truncated:
        answer_ids = (
            answer_ids[
                :max_answer_tokens
            ]
        )

    full_sequence = (
        prompt_ids
        + answer_ids
        + [EOS_TOKEN_ID]
    )

    # input -> target causal shift
    input_ids = (
        full_sequence[:-1]
    )

    labels = (
        full_sequence[1:]
    )

    # Первые позиции относятся к prompt.
    # Первый supervised target находится
    # на позиции len(prompt_ids) - 1.
    first_supervised_position = (
        len(prompt_ids)
        - 1
    )

    labels = [
        (
            token
            if index
            >= first_supervised_position
            else IGNORE_INDEX
        )
        for index, token
        in enumerate(labels)
    ]

    if len(input_ids) > BLOCK_SIZE:
        raise RuntimeError(
            "Internal sequence length "
            "exceeds block size."
        )

    padding = (
        BLOCK_SIZE
        - len(input_ids)
    )

    input_ids.extend(
        [EOS_TOKEN_ID]
        * padding
    )

    labels.extend(
        [IGNORE_INDEX]
        * padding
    )

    target_answer_ids = (
        answer_ids
        + [EOS_TOKEN_ID]
    )

    return {
        "question":
            question,

        "answer":
            answer,

        "prompt":
            prompt,

        "prompt_ids":
            prompt_ids,

        "answer_ids":
            answer_ids,

        "target_answer_ids":
            target_answer_ids,

        "input_ids":
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

        "labels":
            torch.tensor(
                labels,
                dtype=torch.long,
            ),

        "first_supervised_position":
            first_supervised_position,

        "truncated":
            truncated,
    }


# ============================================================================
# CHECKPOINT / MODEL
# ============================================================================

def load_checkpoint(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{path}"
        )

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except Exception:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def load_model(
    path: Path,
):
    checkpoint = (
        load_checkpoint(
            path
        )
    )

    model, tie_weights = (
        build_model_for_checkpoint(
            checkpoint=checkpoint,
            vocab_size=int(
                MODEL_CONFIG[
                    "vocab_size"
                ]
            ),
            embed_dim=int(
                MODEL_CONFIG[
                    "embed_dim"
                ]
            ),
            num_heads=int(
                MODEL_CONFIG[
                    "num_heads"
                ]
            ),
            num_layers=int(
                MODEL_CONFIG[
                    "num_layers"
                ]
            ),
            block_size=int(
                MODEL_CONFIG[
                    "block_size"
                ]
            ),
            dropout=0.0,
            ignore_index=IGNORE_INDEX,
        )
    )

    metadata = {
        "iter_num":
            checkpoint.get(
                "iter_num"
            )
            if isinstance(
                checkpoint,
                dict,
            )
            else None,

        "best_val_loss":
            checkpoint.get(
                "best_val_loss"
            )
            if isinstance(
                checkpoint,
                dict,
            )
            else None,

        "tie_weights":
            tie_weights,
    }

    del checkpoint
    gc.collect()

    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        metadata,
    )


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return nullcontext()


# ============================================================================
# TEACHER-FORCED EVALUATION
# ============================================================================

@torch.inference_mode()
def evaluate_dataset(
    model,
    examples: list[dict],
    label: str,
    split_name: str,
) -> dict:
    total_nll = 0.0
    total_tokens = 0

    correct_tokens = 0

    first_token_correct = 0
    first_token_count = 0

    whole_answer_correct = 0

    started = time.time()

    print(
        f"\n{label} | "
        f"{split_name}: "
        f"{len(examples)} examples"
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):
        input_ids = (
            example[
                "input_ids"
            ]
            .unsqueeze(0)
            .to(DEVICE)
        )

        labels = (
            example[
                "labels"
            ]
            .unsqueeze(0)
            .to(DEVICE)
        )

        with autocast_context():
            logits, _ = model(
                input_ids
            )

        mask = (
            labels
            != IGNORE_INDEX
        )

        active_logits = (
            logits[
                mask
            ]
            .float()
        )

        active_targets = (
            labels[
                mask
            ]
        )

        if active_targets.numel() == 0:
            raise RuntimeError(
                "Example contains no "
                "supervised tokens."
            )

        nll = F.cross_entropy(
            active_logits,
            active_targets,
            reduction="sum",
        )

        predictions = (
            active_logits.argmax(
                dim=-1
            )
        )

        matches = (
            predictions
            == active_targets
        )

        total_nll += float(
            nll.item()
        )

        total_tokens += int(
            active_targets.numel()
        )

        correct_tokens += int(
            matches.sum().item()
        )

        if bool(
            matches.all().item()
        ):
            whole_answer_correct += 1

        first_position = int(
            example[
                "first_supervised_position"
            ]
        )

        first_logits = (
            logits[
                0,
                first_position,
                :,
            ]
            .float()
        )

        first_target = int(
            labels[
                0,
                first_position,
            ].item()
        )

        first_prediction = int(
            first_logits.argmax().item()
        )

        first_token_count += 1

        if (
            first_prediction
            == first_target
        ):
            first_token_correct += 1

        if (
            index % 100 == 0
            or index
            == len(examples)
        ):
            print(
                f"  {index:4d}/"
                f"{len(examples):4d}"
            )

        del (
            input_ids,
            labels,
            logits,
            mask,
            active_logits,
            active_targets,
            predictions,
            matches,
            nll,
        )

    elapsed = (
        time.time()
        - started
    )

    mean_nll = (
        total_nll
        / total_tokens
    )

    perplexity = math.exp(
        min(
            mean_nll,
            20.0,
        )
    )

    token_accuracy = (
        correct_tokens
        / total_tokens
    )

    first_accuracy = (
        first_token_correct
        / first_token_count
    )

    whole_accuracy = (
        whole_answer_correct
        / len(examples)
    )

    result = {
        "loss":
            mean_nll,

        "perplexity":
            perplexity,

        "token_accuracy":
            token_accuracy,

        "first_token_accuracy":
            first_accuracy,

        "whole_answer_accuracy":
            whole_accuracy,

        "supervised_tokens":
            total_tokens,

        "examples":
            len(examples),

        "seconds":
            elapsed,
    }

    print(
        f"  loss:                "
        f"{mean_nll:.6f}"
    )

    print(
        f"  perplexity:          "
        f"{perplexity:.4f}"
    )

    print(
        f"  token accuracy:      "
        f"{token_accuracy:.2%}"
    )

    print(
        f"  first-token accuracy:"
        f" {first_accuracy:.2%}"
    )

    print(
        f"  whole answer exact:  "
        f"{whole_accuracy:.2%}"
    )

    return result


# ============================================================================
# AUTOREGRESSIVE GREEDY ROLLOUT
# ============================================================================

@torch.inference_mode()
def greedy_rollout(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    new_tokens: list[int] = []

    for _ in range(
        max_new_tokens
    ):
        idx_cond = (
            generated[
                :,
                -BLOCK_SIZE:,
            ]
        )

        with autocast_context():
            logits, _ = model(
                idx_cond
            )

        next_token = int(
            logits[
                0,
                -1,
                :,
            ]
            .argmax()
            .item()
        )

        new_tokens.append(
            next_token
        )

        next_tensor = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=DEVICE,
        )

        generated = torch.cat(
            [
                generated,
                next_tensor,
            ],
            dim=1,
        )

        if (
            next_token
            == EOS_TOKEN_ID
        ):
            break

    return new_tokens


def common_prefix_length(
    generated: list[int],
    target: list[int],
) -> int:
    count = 0

    for generated_token, target_token in zip(
        generated,
        target,
    ):
        if (
            generated_token
            != target_token
        ):
            break

        count += 1

    return count


@torch.inference_mode()
def rollout_samples(
    model,
    examples: list[dict],
    sample_indices: list[int],
) -> list[dict]:
    results = []

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):
        example = (
            examples[index]
        )

        target = (
            example[
                "target_answer_ids"
            ]
        )

        max_tokens = min(
            ROLLOUT_MAX_TOKENS,
            max(
                1,
                len(target),
            ),
        )

        generated = (
            greedy_rollout(
                model=model,
                prompt_ids=(
                    example[
                        "prompt_ids"
                    ]
                ),
                max_new_tokens=(
                    max_tokens
                ),
            )
        )

        prefix = (
            common_prefix_length(
                generated,
                target,
            )
        )

        if (
            prefix
            == len(target)
            and len(generated)
            >= len(target)
        ):
            divergence = (
                "NO DIVERGENCE "
                "WITHIN TARGET"
            )

        elif (
            prefix
            >= len(generated)
        ):
            divergence = (
                "NO DIVERGENCE "
                "WITHIN GENERATED TOKENS"
            )

        else:
            divergence = (
                f"token #{prefix + 1}"
            )

        results.append(
            {
                "dataset_index":
                    index,

                "question":
                    example[
                        "question"
                    ],

                "reference_answer":
                    example[
                        "answer"
                    ],

                "generated_tokens":
                    generated,

                "generated_text":
                    decode_tokens(
                        [
                            token
                            for token
                            in generated
                            if token
                            != EOS_TOKEN_ID
                        ]
                    ).strip(),

                "target_tokens":
                    len(target),

                "common_prefix":
                    prefix,

                "divergence":
                    divergence,

                "first_token_correct":
                    bool(
                        generated
                        and target
                        and generated[0]
                        == target[0]
                    ),
            }
        )

        print(
            f"  rollout "
            f"{number}/"
            f"{len(sample_indices)} "
            f"| train index {index}"
        )

    return results


# ============================================================================
# HELPERS
# ============================================================================

def select_sample_indices(
    total: int,
    count: int,
) -> list[int]:
    if total <= 0:
        return []

    if count <= 1:
        return [0]

    if total <= count:
        return list(
            range(total)
        )

    return [
        round(
            index
            * (total - 1)
            / (count - 1)
        )
        for index
        in range(count)
    ]


def free_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_metrics(
    name: str,
    metrics: dict,
) -> list[str]:
    return [
        f"{name}",
        (
            f"  examples:             "
            f"{metrics['examples']}"
        ),
        (
            f"  supervised tokens:    "
            f"{metrics['supervised_tokens']}"
        ),
        (
            f"  token-weighted loss:  "
            f"{metrics['loss']:.6f}"
        ),
        (
            f"  perplexity:           "
            f"{metrics['perplexity']:.4f}"
        ),
        (
            f"  token accuracy:       "
            f"{metrics['token_accuracy']:.2%}"
        ),
        (
            f"  first-token accuracy: "
            f"{metrics['first_token_accuracy']:.2%}"
        ),
        (
            f"  whole-answer exact:   "
            f"{metrics['whole_answer_accuracy']:.2%}"
        ),
        (
            f"  time:                 "
            f"{metrics['seconds']:.2f}s"
        ),
    ]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 100)
    print(
        "NOVA_v1 — SFT TEACHER FORCING "
        "vs AUTOREGRESSIVE ROLLOUT"
    )
    print("=" * 100)

    print(
        f"Device:    {DEVICE}"
    )

    print(
        f"FINAL:     "
        f"{FINAL_CHECKPOINT}"
    )

    print(
        f"SFT-120:   "
        f"{SFT_CHECKPOINT}"
    )

    print(
        f"Train:     "
        f"{TRAIN_PATH}"
    )

    print(
        f"Val:       "
        f"{VAL_PATH}"
    )

    train_records = (
        read_jsonl(
            TRAIN_PATH
        )
    )

    val_records = (
        read_jsonl(
            VAL_PATH
        )
    )

    train_examples = [
        build_example(record)
        for record
        in train_records
    ]

    val_examples = [
        build_example(record)
        for record
        in val_records
    ]

    print(
        f"Train examples: "
        f"{len(train_examples)}"
    )

    print(
        f"Val examples:   "
        f"{len(val_examples)}"
    )

    truncated_train = sum(
        int(
            example[
                "truncated"
            ]
        )
        for example
        in train_examples
    )

    truncated_val = sum(
        int(
            example[
                "truncated"
            ]
        )
        for example
        in val_examples
    )

    print(
        f"Train truncated:"
        f" {truncated_train}"
    )

    print(
        f"Val truncated:  "
        f" {truncated_val}"
    )

    sample_indices = (
        select_sample_indices(
            total=len(
                train_examples
            ),
            count=(
                ROLLOUT_SAMPLE_COUNT
            ),
        )
    )

    model_results = {}

    for label, path in [
        (
            "FINAL",
            FINAL_CHECKPOINT,
        ),
        (
            "SFT-120",
            SFT_CHECKPOINT,
        ),
    ]:
        print()
        print("=" * 100)
        print(
            f"LOADING {label}"
        )
        print("=" * 100)

        model, metadata = (
            load_model(
                path
            )
        )

        print(
            "Architecture: "
            + (
                "TIED"
                if metadata[
                    "tie_weights"
                ]
                else "UNTIED"
            )
        )

        print(
            f"iter_num: "
            f"{metadata['iter_num']}"
        )

        print(
            f"best_val_loss: "
            f"{metadata['best_val_loss']}"
        )

        train_metrics = (
            evaluate_dataset(
                model=model,
                examples=(
                    train_examples
                ),
                label=label,
                split_name="TRAIN",
            )
        )

        val_metrics = (
            evaluate_dataset(
                model=model,
                examples=(
                    val_examples
                ),
                label=label,
                split_name="VAL",
            )
        )

        print(
            f"\n{label} | "
            f"GREEDY EXACT-TRAIN "
            f"ROLLOUTS"
        )

        rollouts = (
            rollout_samples(
                model=model,
                examples=(
                    train_examples
                ),
                sample_indices=(
                    sample_indices
                ),
            )
        )

        model_results[
            label
        ] = {
            "metadata":
                metadata,

            "train":
                train_metrics,

            "val":
                val_metrics,

            "rollouts":
                rollouts,
        }

        del model
        free_cuda()

    # ========================================================================
    # REPORT
    # ========================================================================

    lines: list[str] = [
        "=" * 100,
        (
            "NOVA_v1 — SFT TEACHER FORCING "
            "vs AUTOREGRESSIVE ROLLOUT"
        ),
        "=" * 100,
        "",
        f"FINAL:   {FINAL_CHECKPOINT}",
        f"SFT-120: {SFT_CHECKPOINT}",
        "",
        (
            f"TRAIN examples: "
            f"{len(train_examples)}"
        ),
        (
            f"VAL examples:   "
            f"{len(val_examples)}"
        ),
        (
            f"TRAIN truncated:"
            f" {truncated_train}"
        ),
        (
            f"VAL truncated:  "
            f"{truncated_val}"
        ),
        "",
        "IMPORTANT:",
        (
            "Token accuracy below is measured "
            "with TEACHER FORCING."
        ),
        (
            "Greedy rollout is measured "
            "AUTOREGRESSIVELY from only the prompt."
        ),
        "",
    ]

    for label in [
        "FINAL",
        "SFT-120",
    ]:
        result = (
            model_results[
                label
            ]
        )

        lines.extend(
            [
                "=" * 100,
                label,
                "=" * 100,
                (
                    f"iter_num: "
                    f"{result['metadata']['iter_num']}"
                ),
                (
                    f"checkpoint best_val_loss: "
                    f"{result['metadata']['best_val_loss']}"
                ),
                "",
            ]
        )

        lines.extend(
            format_metrics(
                "TRAIN",
                result[
                    "train"
                ],
            )
        )

        lines.append("")

        lines.extend(
            format_metrics(
                "VAL",
                result[
                    "val"
                ],
            )
        )

        lines.append("")

    lines.extend(
        [
            "=" * 100,
            "DIRECT COMPARISON",
            "=" * 100,
            "",
        ]
    )

    for split in [
        "train",
        "val",
    ]:
        final = (
            model_results[
                "FINAL"
            ][split]
        )

        sft = (
            model_results[
                "SFT-120"
            ][split]
        )

        lines.extend(
            [
                split.upper(),
                (
                    "  loss: "
                    f"{final['loss']:.6f}"
                    " -> "
                    f"{sft['loss']:.6f}"
                ),
                (
                    "  token accuracy: "
                    f"{final['token_accuracy']:.2%}"
                    " -> "
                    f"{sft['token_accuracy']:.2%}"
                ),
                (
                    "  first-token accuracy: "
                    f"{final['first_token_accuracy']:.2%}"
                    " -> "
                    f"{sft['first_token_accuracy']:.2%}"
                ),
                (
                    "  whole-answer exact: "
                    f"{final['whole_answer_accuracy']:.2%}"
                    " -> "
                    f"{sft['whole_answer_accuracy']:.2%}"
                ),
                "",
            ]
        )

    final_rollouts = {
        item[
            "dataset_index"
        ]: item
        for item
        in model_results[
            "FINAL"
        ][
            "rollouts"
        ]
    }

    sft_rollouts = {
        item[
            "dataset_index"
        ]: item
        for item
        in model_results[
            "SFT-120"
        ][
            "rollouts"
        ]
    }

    lines.extend(
        [
            "=" * 100,
            (
                "GREEDY AUTOREGRESSIVE "
                "ROLLOUTS ON EXACT TRAIN EXAMPLES"
            ),
            "=" * 100,
            "",
        ]
    )

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):
        final = (
            final_rollouts[
                index
            ]
        )

        sft = (
            sft_rollouts[
                index
            ]
        )

        lines.extend(
            [
                "-" * 100,
                (
                    f"SAMPLE {number} "
                    f"| TRAIN INDEX {index}"
                ),
                "-" * 100,
                "",
                "QUESTION:",
                final[
                    "question"
                ],
                "",
                "REFERENCE ANSWER:",
                final[
                    "reference_answer"
                ],
                "",
                "FINAL GREEDY:",
                (
                    final[
                        "generated_text"
                    ]
                    or "(empty)"
                ),
                (
                    "FINAL first token correct: "
                    f"{final['first_token_correct']}"
                ),
                (
                    "FINAL correct target prefix: "
                    f"{final['common_prefix']} tokens"
                ),
                (
                    "FINAL first divergence: "
                    f"{final['divergence']}"
                ),
                "",
                "SFT-120 GREEDY:",
                (
                    sft[
                        "generated_text"
                    ]
                    or "(empty)"
                ),
                (
                    "SFT first token correct: "
                    f"{sft['first_token_correct']}"
                ),
                (
                    "SFT correct target prefix: "
                    f"{sft['common_prefix']} tokens"
                ),
                (
                    "SFT first divergence: "
                    f"{sft['divergence']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "=" * 100,
            "END",
            "=" * 100,
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "DIAGNOSTIC FINISHED"
    )
    print("=" * 100)

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()