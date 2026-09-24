from __future__ import annotations

import sys
import time
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


from config import MODEL_CONFIG, RUNTIME_CONFIG  # noqa: E402
from inference import Assistant  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "decoding_comparison.txt"
)


TESTS = [
    {
        "name": "2 + 2",
        "query": "Сколько будет 2 + 2?",
        "reference": "4",
    },

    {
        "name": "Australia",
        "query": "Какая столица Австралии?",
        "reference": "Канберра",
    },

    {
        "name": "Python",
        "query": "Что такое Python?",
        "reference": (
            "Высокоуровневый язык "
            "программирования общего назначения."
        ),
    },

    {
        "name": "is vs ==",
        "query": (
            "Когда в Python использовать "
            "is, а когда ==?"
        ),
        "reference": (
            "is — идентичность объектов; "
            "== — равенство значений."
        ),
    },

    {
        "name": "TCP vs UDP",
        "query": (
            "Объясни простыми словами "
            "разницу между TCP и UDP."
        ),
        "reference": (
            "TCP обеспечивает надежную "
            "упорядоченную доставку; UDP "
            "не дает таких гарантий."
        ),
    },

    {
        "name": "REST",
        "query": "Что такое REST API?",
        "reference": (
            "REST — архитектурный стиль "
            "для распределенных систем."
        ),
    },

    {
        "name": "Docker vs VM",
        "query": (
            "Чем Docker-контейнер "
            "отличается от виртуальной машины?"
        ),
        "reference": (
            "Контейнер обычно разделяет "
            "ядро host OS; VM имеет "
            "свою guest OS."
        ),
    },

    {
        "name": "Git fetch vs pull",
        "query": (
            "Чем git fetch отличается "
            "от git pull?"
        ),
        "reference": (
            "fetch получает изменения; "
            "pull получает и интегрирует."
        ),
    },

    {
        "name": "PostgreSQL",
        "query": "Что такое PostgreSQL?",
        "reference": (
            "Объектно-реляционная СУБД."
        ),
    },

    {
        "name": "Pandas NaN",
        "query": (
            "Как в pandas заполнить "
            "пропущенные значения NaN?"
        ),
        "reference": (
            "Например fillna(), ffill(), "
            "bfill() — в зависимости "
            "от задачи."
        ),
    },
]


MODES = [
    {
        "name": "GREEDY",
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "seed": None,
    },

    {
        "name": "APP DEFAULT — SEED 1337",
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "seed": 1337,
    },

    {
        "name": "APP DEFAULT — SEED 2026",
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "seed": 2026,
    },
]


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — DECODING COMPARISON — NO RAG"
    )
    print("=" * 100)
    print()

    assistant = Assistant(
        checkpoint_path=(
            RUNTIME_CONFIG[
                "checkpoint"
            ]
        ),
        model_config=MODEL_CONFIG,
        device=(
            RUNTIME_CONFIG[
                "device"
            ]
        ),
        rag_core_index_path=(
            RUNTIME_CONFIG[
                "rag_core_index"
            ]
        ),
        rag_technical_index_path=(
            RUNTIME_CONFIG[
                "rag_technical_index"
            ]
        ),
        rag_broad_index_path=(
            RUNTIME_CONFIG[
                "rag_broad_index"
            ]
        ),
        rag_core_threshold=(
            RUNTIME_CONFIG[
                "rag_core_threshold"
            ]
        ),
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA DECODING COMPARISON"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        f"Checkpoint: "
        f"{RUNTIME_CONFIG['checkpoint']}"
    )

    lines.append(
        "RAG: OFF"
    )

    lines.append(
        f"Questions: {len(TESTS)}"
    )

    lines.append(
        f"Modes per question: "
        f"{len(MODES)}"
    )

    lines.append("")

    total_started = time.time()

    for number, test in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{number:02d}/"
            f"{len(TESTS):02d}] "
            f"{test['name']}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUESTION {number:02d}: "
            f"{test['name']}"
        )

        lines.append(
            "=" * 100
        )

        lines.append("")

        lines.append(
            "QUERY:"
        )

        lines.append(
            test[
                "query"
            ]
        )

        lines.append("")

        lines.append(
            "REFERENCE:"
        )

        lines.append(
            test[
                "reference"
            ]
        )

        lines.append("")

        for mode in MODES:
            started = time.time()

            (
                answer,
                contexts,
                rag_info,
            ) = assistant.generate(
                query=test[
                    "query"
                ],
                max_new_tokens=160,
                temperature=(
                    mode[
                        "temperature"
                    ]
                ),
                top_k=(
                    mode[
                        "top_k"
                    ]
                ),
                top_p=(
                    mode[
                        "top_p"
                    ]
                ),
                repetition_penalty=(
                    mode[
                        "repetition_penalty"
                    ]
                ),
                seed=(
                    mode[
                        "seed"
                    ]
                ),
                rag_enabled=False,
                rag_source="off",
                answer_mode="auto",
                return_rag_info=True,
            )

            elapsed = (
                time.time()
                - started
            )

            lines.append(
                "-" * 100
            )

            lines.append(
                f"MODE: "
                f"{mode['name']}"
            )

            lines.append(
                f"temperature: "
                f"{mode['temperature']}"
            )

            lines.append(
                f"top_k: "
                f"{mode['top_k']}"
            )

            lines.append(
                f"top_p: "
                f"{mode['top_p']}"
            )

            lines.append(
                "repetition_penalty: "
                f"{mode['repetition_penalty']}"
            )

            lines.append(
                f"seed: "
                f"{mode['seed']}"
            )

            lines.append(
                f"time: "
                f"{elapsed:.3f}s"
            )

            lines.append("")

            lines.append(
                "ANSWER:"
            )

            lines.append(
                answer.strip()
                if answer.strip()
                else "(empty)"
            )

            lines.append("")

            lines.append(
                "MANUAL VERDICT: "
                "[NOT REVIEWED]"
            )

            lines.append("")

    total_time = (
        time.time()
        - total_started
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "END"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"TOTAL GENERATIONS: "
        f"{len(TESTS) * len(MODES)}"
    )

    lines.append(
        f"TOTAL TIME: "
        f"{total_time:.2f}s"
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
        "DECODING COMPARISON FINISHED"
    )
    print("=" * 100)

    print(
        f"Generations: "
        f"{len(TESTS) * len(MODES)}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    print()

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()