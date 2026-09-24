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
    / "grounded_answer_path_test.txt"
)


TRUSTED_TESTS = [
    {
        "name": "CORE POSTGRESQL",
        "query": (
            "Postgres — это язык, "
            "база данных или СУБД?"
        ),
        "rag_source": "auto",
        "expected_route": "core",
        "expected_id": (
            "core_postgresql"
        ),
        "must_contain": [
            "PostgreSQL",
            "система управления",
            "SQL",
        ],
    },

    {
        "name": "CORE MUTABLE DEFAULT",
        "query": (
            "Почему список в "
            "default-аргументе Python "
            "сохраняет изменения "
            "между вызовами?"
        ),
        "rag_source": "auto",
        "expected_route": "core",
        "expected_id": (
            "core_python_mutable_default"
        ),
        "must_contain": [
            "один раз",
            "изменяемый объект",
            "None",
        ],
    },

    {
        "name": "CORE REST",
        "query": (
            "REST — это просто HTTP "
            "с JSON или что-то другое?"
        ),
        "rag_source": "auto",
        "expected_route": "core",
        "expected_id": (
            "core_rest_api"
        ),
        "must_contain": [
            "архитектурный стиль",
            "HTTP",
            "JSON",
        ],
    },

    {
        "name": "TECHNICAL DOCKER",
        "query": (
            "Чем docker stop "
            "отличается от docker kill?"
        ),
        "rag_source": "technical",
        "expected_route": (
            "technical"
        ),
        "expected_line": 267,
        "must_contain": [
            "docker stop",
            "docker kill",
            "сигнал",
        ],
    },
]


FALLBACK_TESTS = [
    {
        "name": "AUTO REJECT CUDA",
        "query": (
            "Почему CUDA kernel "
            "получает out of memory?"
        ),
    },

    {
        "name": "AUTO REJECT BORSH",
        "query": (
            "Как приготовить борщ?"
        ),
    },
]


def compact(
    text: str,
    limit: int = 2000,
) -> str:
    text = str(
        text
    ).strip()

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — GROUNDED ANSWER PATH TEST"
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
        "NOVA GROUNDED ANSWER PATH TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Trusted CORE/TECHNICAL:"
    )

    lines.append(
        "  answer_mode=auto"
    )

    lines.append(
        "  expected answer_source=retrieval"
    )

    lines.append("")

    lines.append(
        "Rejected AUTO:"
    )

    lines.append(
        "  expected route=none"
    )

    lines.append(
        "  expected answer_source=model"
    )

    lines.append("")

    trusted_pass = 0
    fallback_pass = 0

    started_all = time.time()

    # ========================================================
    # TRUSTED DIRECT ANSWERS
    # ========================================================

    lines.append(
        "=" * 100
    )

    lines.append(
        "TRUSTED DIRECT ANSWERS"
    )

    lines.append(
        "=" * 100
    )

    for number, test in enumerate(
        TRUSTED_TESTS,
        start=1,
    ):
        print(
            f"[TRUSTED {number}/"
            f"{len(TRUSTED_TESTS)}] "
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
            max_new_tokens=120,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            seed=1337,
            rag_enabled=True,
            rag_source=test[
                "rag_source"
            ],
            answer_mode="auto",
            rag_mode="inline",
            top_k_docs=1,
            min_score=0.80,
            use_reranker=False,
            return_rag_info=True,
        )

        elapsed = (
            time.time()
            - started
        )

        results = rag_info.get(
            "results",
            [],
        )

        metadata = (
            results[0][2]
            if results
            else {}
        )

        actual_id = (
            metadata.get(
                "id"
            )
            if metadata
            else None
        )

        actual_line = (
            metadata.get(
                "line"
            )
            if metadata
            else None
        )

        route_ok = (
            rag_info.get(
                "route"
            )
            == test[
                "expected_route"
            ]
        )

        id_ok = True

        if (
            "expected_id"
            in test
        ):
            id_ok = (
                actual_id
                == test[
                    "expected_id"
                ]
            )

        line_ok = True

        if (
            "expected_line"
            in test
        ):
            line_ok = (
                actual_line
                == test[
                    "expected_line"
                ]
            )

        source_ok = (
            rag_info.get(
                "answer_source"
            )
            == "retrieval"
        )

        direct_ok = (
            rag_info.get(
                "direct_answer"
            )
            is True
        )

        generation_ok = (
            rag_info.get(
                "model_generated"
            )
            is False
        )

        context_exact = (
            len(contexts) == 1
            and answer.strip()
            == contexts[0].strip()
        )

        content_ok = all(
            phrase.lower()
            in answer.lower()
            for phrase
            in test[
                "must_contain"
            ]
        )

        ok = all(
            [
                route_ok,
                id_ok,
                line_ok,
                source_ok,
                direct_ok,
                generation_ok,
                context_exact,
                content_ok,
            ]
        )

        if ok:
            trusted_pass += 1

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"TRUSTED {number}: "
            f"{test['name']}"
        )

        lines.append(
            f"QUERY: "
            f"{test['query']}"
        )

        lines.append(
            f"ROUTE: "
            f"{rag_info.get('route')}"
        )

        lines.append(
            f"TOP SCORE: "
            f"{rag_info.get('top1_score')}"
        )

        lines.append(
            f"ID: "
            f"{actual_id}"
        )

        lines.append(
            f"LINE: "
            f"{actual_line}"
        )

        lines.append(
            "ANSWER SOURCE: "
            f"{rag_info.get('answer_source')}"
        )

        lines.append(
            "DIRECT ANSWER: "
            f"{rag_info.get('direct_answer')}"
        )

        lines.append(
            "MODEL GENERATED: "
            f"{rag_info.get('model_generated')}"
        )

        lines.append(
            f"ANSWER == CONTEXT: "
            f"{context_exact}"
        )

        lines.append(
            f"CONTENT CHECK: "
            f"{content_ok}"
        )

        lines.append(
            f"TIME: "
            f"{elapsed:.3f}s"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

        lines.append("")

        lines.append(
            "ANSWER:"
        )

        lines.append(
            compact(
                answer
            )
        )

        lines.append("")

    # ========================================================
    # MODEL FALLBACK
    # ========================================================

    lines.append(
        "=" * 100
    )

    lines.append(
        "AUTO REJECT -> MODEL FALLBACK"
    )

    lines.append(
        "=" * 100
    )

    for number, test in enumerate(
        FALLBACK_TESTS,
        start=1,
    ):
        print(
            f"[FALLBACK {number}/"
            f"{len(FALLBACK_TESTS)}] "
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
            max_new_tokens=80,
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=None,
            rag_enabled=True,
            rag_source="auto",
            answer_mode="auto",
            rag_mode="inline",
            top_k_docs=1,
            min_score=0.80,
            use_reranker=False,
            return_rag_info=True,
        )

        elapsed = (
            time.time()
            - started
        )

        route_ok = (
            rag_info.get(
                "route"
            )
            == "none"
        )

        source_ok = (
            rag_info.get(
                "answer_source"
            )
            == "model"
        )

        direct_ok = (
            rag_info.get(
                "direct_answer"
            )
            is False
        )

        generation_ok = (
            rag_info.get(
                "model_generated"
            )
            is True
        )

        contexts_ok = (
            contexts == []
        )

        answer_ok = bool(
            answer.strip()
        )

        ok = all(
            [
                route_ok,
                source_ok,
                direct_ok,
                generation_ok,
                contexts_ok,
                answer_ok,
            ]
        )

        if ok:
            fallback_pass += 1

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"FALLBACK {number}: "
            f"{test['name']}"
        )

        lines.append(
            f"QUERY: "
            f"{test['query']}"
        )

        lines.append(
            f"ROUTE: "
            f"{rag_info.get('route')}"
        )

        lines.append(
            f"CORE SCORE: "
            f"{rag_info.get('top1_score')}"
        )

        lines.append(
            "ANSWER SOURCE: "
            f"{rag_info.get('answer_source')}"
        )

        lines.append(
            "DIRECT ANSWER: "
            f"{rag_info.get('direct_answer')}"
        )

        lines.append(
            "MODEL GENERATED: "
            f"{rag_info.get('model_generated')}"
        )

        lines.append(
            f"CONTEXT COUNT: "
            f"{len(contexts)}"
        )

        lines.append(
            f"ANSWER NONEMPTY: "
            f"{answer_ok}"
        )

        lines.append(
            f"TIME: "
            f"{elapsed:.3f}s"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

        lines.append("")

        lines.append(
            "MODEL ANSWER:"
        )

        lines.append(
            compact(
                answer
            )
        )

        lines.append("")

    # ========================================================
    # FORCED GENERATION DIAGNOSTIC
    # ========================================================

    print(
        "[DIAGNOSTIC] "
        "Forced generation on CORE"
    )

    (
        forced_answer,
        forced_contexts,
        forced_info,
    ) = assistant.generate(
        query=(
            "Что такое PostgreSQL?"
        ),
        max_new_tokens=80,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        seed=None,
        rag_enabled=True,
        rag_source="auto",
        answer_mode="generate",
        rag_mode="inline",
        top_k_docs=1,
        min_score=0.80,
        use_reranker=False,
        return_rag_info=True,
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "DIAGNOSTIC — FORCED GENERATION"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "QUERY: Что такое PostgreSQL?"
    )

    lines.append(
        "answer_mode=generate"
    )

    lines.append(
        f"ROUTE: "
        f"{forced_info.get('route')}"
    )

    lines.append(
        "ANSWER SOURCE: "
        f"{forced_info.get('answer_source')}"
    )

    lines.append(
        "DIRECT ANSWER: "
        f"{forced_info.get('direct_answer')}"
    )

    lines.append(
        "MODEL GENERATED: "
        f"{forced_info.get('model_generated')}"
    )

    lines.append(
        f"CONTEXT COUNT: "
        f"{len(forced_contexts)}"
    )

    lines.append("")

    lines.append(
        "ANSWER:"
    )

    lines.append(
        compact(
            forced_answer
        )
    )

    lines.append("")

    # ========================================================
    # SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - started_all
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "SUMMARY"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "TRUSTED DIRECT: "
        f"{trusted_pass}/"
        f"{len(TRUSTED_TESTS)}"
    )

    lines.append(
        "MODEL FALLBACK: "
        f"{fallback_pass}/"
        f"{len(FALLBACK_TESTS)}"
    )

    lines.append(
        f"TOTAL TIME: "
        f"{total_time:.2f}s"
    )

    lines.append(
        "=" * 100
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
        "TRUSTED DIRECT: "
        f"{trusted_pass}/"
        f"{len(TRUSTED_TESTS)}"
    )

    print(
        "MODEL FALLBACK: "
        f"{fallback_pass}/"
        f"{len(FALLBACK_TESTS)}"
    )

    print()

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "GROUNDED ANSWER PATH "
        "TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()