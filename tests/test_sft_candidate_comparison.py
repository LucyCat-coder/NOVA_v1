from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch


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


FINAL_CHECKPOINT = (
    PROJECT_ROOT
    / "out"
    / "final"
    / "best_final.pt"
)

CANDIDATE_CHECKPOINT = (
    PROJECT_ROOT
    / "out"
    / "candidate_sft"
    / "technical_v1"
    / "best_candidate.pt"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sft_candidate_comparison.txt"
)


TESTS = [
    # ========================================================
    # EXACT / NEAR-EXACT TECHNICAL
    # ========================================================

    {
        "group": "TECH_EXACT",
        "name": "is vs ==",
        "query": (
            "Когда в Python применять is, "
            "а когда ==?"
        ),
        "reference": (
            "== сравнивает значения объектов, "
            "а is проверяет идентичность. "
            "Для None обычно используют is."
        ),
    },

    {
        "group": "TECH_EXACT",
        "name": "Closure",
        "query": (
            "Что такое замыкание в Python? "
            "Покажи практический короткий пример."
        ),
        "reference": (
            "Внутренняя функция сохраняет доступ "
            "к переменным внешней области."
        ),
    },

    {
        "group": "TECH_EXACT",
        "name": "Docker stop vs kill",
        "query": (
            "Чем docker stop отличается "
            "от docker kill?"
        ),
        "reference": (
            "stop пытается завершить процесс "
            "корректно и ждёт; kill по умолчанию "
            "завершает немедленно."
        ),
    },

    {
        "group": "TECH_EXACT",
        "name": "COUNT NULL",
        "query": (
            "Таблица содержит значения score: "
            "10, NULL, 20. Что вернут COUNT(*), "
            "COUNT(score) и AVG(score)?"
        ),
        "reference": (
            "COUNT(*) = 3, COUNT(score) = 2, "
            "AVG(score) = 15."
        ),
    },

    # ========================================================
    # PARAPHRASED TECHNICAL
    # ========================================================

    {
        "group": "TECH_PARAPHRASE",
        "name": "is vs == paraphrase",
        "query": (
            "Объясни разницу между операторами "
            "is и == в Python."
        ),
        "reference": (
            "is — идентичность объектов; "
            "== — равенство значений."
        ),
    },

    {
        "group": "TECH_PARAPHRASE",
        "name": "Docker paraphrase",
        "query": (
            "Что произойдёт при docker stop "
            "и чем это отличается от "
            "принудительного docker kill?"
        ),
        "reference": (
            "stop даёт процессу время завершиться "
            "корректно; kill обычно завершает сразу."
        ),
    },

    {
        "group": "TECH_PARAPHRASE",
        "name": "Tuple vs list",
        "query": (
            "В каких случаях в Python лучше "
            "выбрать tuple вместо list?"
        ),
        "reference": (
            "Tuple подходит для фиксированной "
            "неизменяемой структуры; list — "
            "для изменяемой последовательности."
        ),
    },

    # ========================================================
    # GENERAL / RETENTION
    # ========================================================

    {
        "group": "GENERAL",
        "name": "2 + 2",
        "query": (
            "Сколько будет 2 + 2?"
        ),
        "reference": (
            "4"
        ),
    },

    {
        "group": "GENERAL",
        "name": "Australia",
        "query": (
            "Какая столица Австралии?"
        ),
        "reference": (
            "Канберра."
        ),
    },

    {
        "group": "GENERAL",
        "name": "Python",
        "query": (
            "Что такое Python?"
        ),
        "reference": (
            "Высокоуровневый язык "
            "программирования общего назначения."
        ),
    },

    {
        "group": "GENERAL",
        "name": "PostgreSQL",
        "query": (
            "Что такое PostgreSQL?"
        ),
        "reference": (
            "Свободная объектно-реляционная СУБД."
        ),
    },

    {
        "group": "GENERAL",
        "name": "TCP vs UDP",
        "query": (
            "Объясни простыми словами "
            "разницу между TCP и UDP."
        ),
        "reference": (
            "TCP обеспечивает надежную "
            "упорядоченную доставку; "
            "UDP не гарантирует её."
        ),
    },

    {
        "group": "GENERAL",
        "name": "REST",
        "query": (
            "Что такое REST API?"
        ),
        "reference": (
            "REST — архитектурный стиль "
            "для распределённых систем."
        ),
    },

    {
        "group": "GENERAL",
        "name": "Pandas NaN",
        "query": (
            "Как в pandas заполнить "
            "пропущенные значения NaN?"
        ),
        "reference": (
            "Например fillna(), ffill(), "
            "bfill() — в зависимости от задачи."
        ),
    },
]


def load_assistant(
    checkpoint_path: Path,
) -> Assistant:
    return Assistant(
        checkpoint_path=checkpoint_path,
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


def run_model(
    *,
    label: str,
    checkpoint_path: Path,
) -> list[dict]:
    print()
    print("=" * 100)
    print(
        f"LOADING: {label}"
    )
    print("=" * 100)

    assistant = (
        load_assistant(
            checkpoint_path
        )
    )

    results = []

    for number, test in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{number:02d}/"
            f"{len(TESTS):02d}] "
            f"{test['group']} | "
            f"{test['name']}"
        )

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

            # Детерминированное сравнение.
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=None,

            # Только checkpoint.
            rag_enabled=False,
            rag_source="off",
            answer_mode="auto",

            return_rag_info=True,
        )

        elapsed = (
            time.time()
            - started
        )

        results.append(
            {
                "group":
                    test[
                        "group"
                    ],

                "name":
                    test[
                        "name"
                    ],

                "query":
                    test[
                        "query"
                    ],

                "reference":
                    test[
                        "reference"
                    ],

                "answer":
                    str(
                        answer
                    ).strip(),

                "time":
                    elapsed,

                "contexts":
                    len(
                        contexts
                    ),

                "route":
                    rag_info.get(
                        "route"
                    ),

                "answer_source":
                    rag_info.get(
                        "answer_source"
                    ),
            }
        )

    del assistant

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — FINAL vs 20-STEP SFT CANDIDATE"
    )
    print("=" * 100)
    print()

    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"FINAL checkpoint не найден: "
            f"{FINAL_CHECKPOINT}"
        )

    if not CANDIDATE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Candidate checkpoint не найден: "
            f"{CANDIDATE_CHECKPOINT}"
        )

    print(
        f"FINAL:     "
        f"{FINAL_CHECKPOINT}"
    )

    print(
        f"CANDIDATE: "
        f"{CANDIDATE_CHECKPOINT}"
    )

    print(
        f"Questions: "
        f"{len(TESTS)}"
    )

    print()

    final_results = (
        run_model(
            label="FINAL",
            checkpoint_path=(
                FINAL_CHECKPOINT
            ),
        )
    )

    candidate_results = (
        run_model(
            label="20-STEP SFT CANDIDATE",
            checkpoint_path=(
                CANDIDATE_CHECKPOINT
            ),
        )
    )

    if (
        len(final_results)
        != len(candidate_results)
    ):
        raise AssertionError(
            "Result count mismatch."
        )

    lines = [
        "=" * 100,
        "NOVA — FINAL vs 20-STEP SFT CANDIDATE",
        "=" * 100,
        "",
        f"FINAL checkpoint: "
        f"{FINAL_CHECKPOINT}",
        f"CANDIDATE checkpoint: "
        f"{CANDIDATE_CHECKPOINT}",
        "",
        "Generation:",
        "  RAG = OFF",
        "  temperature = 0",
        "  top_k = OFF",
        "  top_p = OFF",
        "  repetition_penalty = 1",
        "  max_new_tokens = 160",
        "",
    ]

    for index, (
        final_result,
        candidate_result,
    ) in enumerate(
        zip(
            final_results,
            candidate_results,
        ),
        start=1,
    ):
        if (
            final_result[
                "query"
            ]
            != candidate_result[
                "query"
            ]
        ):
            raise AssertionError(
                "Query mismatch."
            )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUESTION {index:02d}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"GROUP: "
            f"{final_result['group']}"
        )

        lines.append(
            f"NAME: "
            f"{final_result['name']}"
        )

        lines.append("")

        lines.append(
            "QUERY:"
        )

        lines.append(
            final_result[
                "query"
            ]
        )

        lines.append("")

        lines.append(
            "REFERENCE:"
        )

        lines.append(
            final_result[
                "reference"
            ]
        )

        lines.append("")

        lines.append(
            "-" * 100
        )

        lines.append(
            "FINAL ANSWER:"
        )

        lines.append(
            final_result[
                "answer"
            ]
            or "(empty)"
        )

        lines.append("")

        lines.append(
            f"FINAL TIME: "
            f"{final_result['time']:.3f}s"
        )

        lines.append("")

        lines.append(
            "-" * 100
        )

        lines.append(
            "CANDIDATE ANSWER:"
        )

        lines.append(
            candidate_result[
                "answer"
            ]
            or "(empty)"
        )

        lines.append("")

        lines.append(
            f"CANDIDATE TIME: "
            f"{candidate_result['time']:.3f}s"
        )

        lines.append("")

        # Намеренно не оцениваем автоматически.
        lines.append(
            "MANUAL VERDICT:"
        )

        lines.append(
            "  FINAL:     [NOT REVIEWED]"
        )

        lines.append(
            "  CANDIDATE: [NOT REVIEWED]"
        )

        lines.append("")

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
        "COMPARISON FINISHED"
    )
    print("=" * 100)

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()