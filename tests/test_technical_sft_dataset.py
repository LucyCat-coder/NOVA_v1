from __future__ import annotations

import sys
from pathlib import Path


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


from config.training.technical_config import (  # noqa: E402
    TECHNICAL_SFT_CONFIG,
)
from train.technical_dataset import (  # noqa: E402
    TechnicalSFTDataset,
)


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "technical_sft_dataset_test.txt"
)


def inspect_dataset(
    *,
    name: str,
    dataset: TechnicalSFTDataset,
) -> list[str]:
    lines = []

    truncated_count = 0

    min_supervised = None
    max_supervised = 0

    total_supervised = 0

    for index in range(
        len(dataset)
    ):
        example = (
            dataset.encode_example(
                index
            )
        )

        input_ids = (
            example.input_ids
        )

        labels = (
            example.labels
        )

        if (
            len(input_ids)
            != dataset.block_size
        ):
            raise AssertionError(
                f"{name} index={index}: "
                "input length != block_size"
            )

        if (
            len(labels)
            != dataset.block_size
        ):
            raise AssertionError(
                f"{name} index={index}: "
                "label length != block_size"
            )

        # ================================================
        # RAW CAUSAL SHIFT
        # ================================================

        raw_sequence = list(
            example.raw_sequence_ids
        )

        raw_input = (
            raw_sequence[:-1]
        )

        raw_targets = (
            raw_sequence[1:]
        )

        actual_raw_input = (
            input_ids[
                :len(
                    raw_input
                )
            ]
            .tolist()
        )

        if (
            actual_raw_input
            != raw_input
        ):
            raise AssertionError(
                f"{name} index={index}: "
                "causal input shift broken"
            )

        # ================================================
        # ASSISTANT-ONLY MASK
        # ================================================

        first_supervised = (
            example.prompt_token_count
            - 1
        )

        if first_supervised < 0:
            raise AssertionError(
                f"{name} index={index}: "
                "invalid prompt token count"
            )

        # Всё до первого assistant prediction
        # должно быть -100.
        if first_supervised > 0:
            prefix_labels = (
                labels[
                    :first_supervised
                ]
            )

            if not bool(
                (
                    prefix_labels
                    == dataset.ignore_index
                )
                .all()
            ):
                raise AssertionError(
                    f"{name} index={index}: "
                    "user/prompt labels "
                    "не полностью masked"
                )

        # Первый assistant prediction
        # обязан быть supervised.
        if (
            int(
                labels[
                    first_supervised
                ].item()
            )
            == dataset.ignore_index
        ):
            raise AssertionError(
                f"{name} index={index}: "
                "первый assistant token "
                "случайно masked"
            )

        # Проверяем сами shifted targets
        # на supervised части.
        for position in range(
            first_supervised,
            len(
                raw_targets
            ),
        ):
            expected = (
                raw_targets[
                    position
                ]
            )

            actual = int(
                labels[
                    position
                ].item()
            )

            if actual != expected:
                raise AssertionError(
                    f"{name} index={index}: "
                    "supervised causal shift "
                    f"broken at position "
                    f"{position}: "
                    f"{actual} != {expected}"
                )

        # ================================================
        # REAL EOS
        # ================================================

        supervised_positions = (
            labels
            != dataset.ignore_index
        ).nonzero(
            as_tuple=False
        ).flatten()

        if len(
            supervised_positions
        ) == 0:
            raise AssertionError(
                f"{name} index={index}: "
                "нет supervised tokens"
            )

        last_supervised_position = int(
            supervised_positions[
                -1
            ].item()
        )

        last_target = int(
            labels[
                last_supervised_position
            ].item()
        )

        if (
            last_target
            != dataset.eos_token_id
        ):
            raise AssertionError(
                f"{name} index={index}: "
                "последний supervised "
                "target не EOS"
            )

        # ================================================
        # PADDING
        # ================================================

        raw_label_count = (
            len(
                raw_targets
            )
        )

        if (
            raw_label_count
            < dataset.block_size
        ):
            padding_labels = (
                labels[
                    raw_label_count:
                ]
            )

            if not bool(
                (
                    padding_labels
                    == dataset.ignore_index
                )
                .all()
            ):
                raise AssertionError(
                    f"{name} index={index}: "
                    "padding labels "
                    "не masked"
                )

        supervised_count = (
            example
            .supervised_token_count
        )

        expected_supervised = (
            example.answer_token_count
            + 1
        )

        if (
            supervised_count
            != expected_supervised
        ):
            raise AssertionError(
                f"{name} index={index}: "
                f"supervised={supervised_count}, "
                f"expected="
                f"{expected_supervised}"
            )

        if example.truncated:
            truncated_count += 1

        total_supervised += (
            supervised_count
        )

        max_supervised = max(
            max_supervised,
            supervised_count,
        )

        if min_supervised is None:
            min_supervised = (
                supervised_count
            )

        else:
            min_supervised = min(
                min_supervised,
                supervised_count,
            )

    average_supervised = (
        total_supervised
        / len(dataset)
    )

    lines.append(
        f"{name} records: "
        f"{len(dataset):,}"
    )

    lines.append(
        f"{name} truncated: "
        f"{truncated_count:,}"
    )

    lines.append(
        f"{name} supervised tokens min: "
        f"{min_supervised}"
    )

    lines.append(
        f"{name} supervised tokens max: "
        f"{max_supervised}"
    )

    lines.append(
        f"{name} supervised tokens mean: "
        f"{average_supervised:.2f}"
    )

    lines.append(
        f"{name} RESULT: PASS"
    )

    return lines


def add_examples(
    *,
    lines: list[str],
    name: str,
    dataset: TechnicalSFTDataset,
):
    indexes = sorted(
        set(
            [
                0,
                len(dataset) // 2,
                len(dataset) - 1,
            ]
        )
    )

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        f"{name} SAMPLE EXAMPLES"
    )

    lines.append(
        "=" * 100
    )

    for index in indexes:
        example = (
            dataset.encode_example(
                index
            )
        )

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"INDEX: {index}"
        )

        lines.append(
            f"SOURCE LINE: "
            f"{example.source_line}"
        )

        lines.append(
            f"PROMPT TOKENS: "
            f"{example.prompt_token_count}"
        )

        lines.append(
            f"ANSWER TOKENS: "
            f"{example.answer_token_count}"
        )

        lines.append(
            f"SUPERVISED TOKENS: "
            f"{example.supervised_token_count}"
        )

        lines.append(
            f"TRUNCATED: "
            f"{example.truncated}"
        )

        lines.append("")

        lines.append(
            "PROMPT:"
        )

        lines.append(
            example.prompt
        )

        lines.append("")

        lines.append(
            "ANSWER:"
        )

        lines.append(
            example.answer
        )


def main():
    config = dict(
        TECHNICAL_SFT_CONFIG
    )

    print()
    print("=" * 100)
    print(
        "NOVA — TECHNICAL SFT DATASET TEST"
    )
    print("=" * 100)
    print()

    train_dataset = (
        TechnicalSFTDataset(
            config[
                "train_data_path"
            ],
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
            config[
                "val_data_path"
            ],
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
        f"Train records: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val records:   "
        f"{len(val_dataset):,}"
    )

    print()

    print(
        "Checking every record..."
    )

    lines = [
        "=" * 100,
        "NOVA TECHNICAL SFT DATASET TEST",
        "=" * 100,
        "",
        f"Train path: "
        f"{config['train_data_path']}",
        f"Val path: "
        f"{config['val_data_path']}",
        f"Block size: "
        f"{config['block_size']}",
        f"EOS token: "
        f"{config['eos_token_id']}",
        f"Ignore index: "
        f"{config['ignore_index']}",
        "",
    ]

    lines.extend(
        inspect_dataset(
            name="TRAIN",
            dataset=train_dataset,
        )
    )

    lines.append("")

    lines.extend(
        inspect_dataset(
            name="VAL",
            dataset=val_dataset,
        )
    )

    add_examples(
        lines=lines,
        name="TRAIN",
        dataset=train_dataset,
    )

    add_examples(
        lines=lines,
        name="VAL",
        dataset=val_dataset,
    )

    lines.extend(
        [
            "",
            "=" * 100,
            "FINAL RESULT: PASS",
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
    print(
        "TRAIN: PASS"
    )

    print(
        "VAL:   PASS"
    )

    print()

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "DATASET TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()