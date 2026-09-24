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
    / "assistant_safe_rag_test.txt"
)


TESTS = [
    {
        "name": "AUTO CORE POSTGRESQL",
        "query": (
            "Postgres — это язык, "
            "база данных или СУБД?"
        ),
        "rag_source": "auto",
        "expected_route": "core",
        "expected_id": (
            "core_postgresql"
        ),
    },

    {
        "name": (
            "AUTO CORE MUTABLE DEFAULT"
        ),
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
    },

    {
        "name": "AUTO CORE REST",
        "query": (
            "REST — это просто HTTP "
            "с JSON или что-то другое?"
        ),
        "rag_source": "auto",
        "expected_route": "core",
        "expected_id": (
            "core_rest_api"
        ),
    },

    {
        "name": "AUTO REJECT CUDA",
        "query": (
            "Почему CUDA kernel "
            "получает out of memory?"
        ),
        "rag_source": "auto",
        "expected_route": "none",
        "expected_id": None,
    },

    {
        "name": "AUTO REJECT BORSH",
        "query": (
            "Как приготовить борщ?"
        ),
        "rag_source": "auto",
        "expected_route": "none",
        "expected_id": None,
    },

    {
        "name": "MANUAL TECHNICAL",
        "query": (
            "Чем docker stop "
            "отличается от docker kill?"
        ),
        "rag_source": "technical",
        "expected_route": (
            "technical"
        ),
        "expected_line": 267,
    },

    {
        "name": "MANUAL BROAD",
        "query": (
            "Как заполнить "
            "пропущенные значения "
            "в pandas DataFrame?"
        ),
        "rag_source": "broad",
        "expected_route": "broad",
    },
]


def compact(
    text: str,
    limit: int = 1000,
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
        "NOVA — ASSISTANT SAFE RAG "
        "END-TO-END TEST"
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
        "NOVA ASSISTANT SAFE RAG "
        "END-TO-END TEST"
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
        "CORE threshold: "
        f"{RUNTIME_CONFIG['rag_core_threshold']}"
    )

    lines.append("")

    route_pass = 0
    generation_nonempty = 0

    started_all = time.time()

    for number, test in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{number}/{len(TESTS)}] "
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

        route = rag_info.get(
            "route"
        )

        results = rag_info.get(
            "results",
            [],
        )

        top_metadata = (
            results[0][2]
            if results
            else {}
        )

        actual_id = (
            top_metadata.get(
                "id"
            )
            if top_metadata
            else None
        )

        actual_line = (
            top_metadata.get(
                "line"
            )
            if top_metadata
            else None
        )

        route_ok = (
            route
            == test[
                "expected_route"
            ]
        )

        if (
            "expected_id"
            in test
        ):
            route_ok = (
                route_ok
                and actual_id
                == test[
                    "expected_id"
                ]
            )

        if (
            "expected_line"
            in test
        ):
            route_ok = (
                route_ok
                and actual_line
                == test[
                    "expected_line"
                ]
            )

        if route_ok:
            route_pass += 1

        answer_nonempty = bool(
            answer.strip()
        )

        if answer_nonempty:
            generation_nonempty += 1

        lines.append(
            "=" * 100
        )

        lines.append(
            f"TEST {number}: "
            f"{test['name']}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUERY: "
            f"{test['query']}"
        )

        lines.append(
            f"RAG SOURCE: "
            f"{test['rag_source']}"
        )

        lines.append(
            f"ROUTE: {route}"
        )

        lines.append(
            f"ACCEPTED: "
            f"{rag_info.get('accepted')}"
        )

        lines.append(
            f"TOP1 SCORE: "
            f"{rag_info.get('top1_score')}"
        )

        lines.append(
            f"TOP ID: "
            f"{actual_id}"
        )

        lines.append(
            f"TOP LINE: "
            f"{actual_line}"
        )

        lines.append(
            "ROUTE RESULT: "
            + (
                "PASS"
                if route_ok
                else "FAIL"
            )
        )

        lines.append(
            "ANSWER NONEMPTY: "
            + (
                "YES"
                if answer_nonempty
                else "NO"
            )
        )

        lines.append(
            f"TIME: "
            f"{elapsed:.3f}s"
        )

        lines.append("")

        lines.append(
            "CONTEXT:"
        )

        if contexts:
            for context_number, context in (
                enumerate(
                    contexts,
                    start=1,
                )
            ):
                lines.append(
                    f"[{context_number}] "
                    + compact(
                        context,
                        1200,
                    )
                )

        else:
            lines.append(
                "(no context)"
            )

        lines.append("")

        lines.append(
            "NOVA ANSWER:"
        )

        lines.append(
            compact(
                answer,
                2000,
            )
        )

        lines.append("")

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
        "ROUTING: "
        f"{route_pass}/"
        f"{len(TESTS)}"
    )

    lines.append(
        "NONEMPTY GENERATION: "
        f"{generation_nonempty}/"
        f"{len(TESTS)}"
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
        "ROUTING: "
        f"{route_pass}/"
        f"{len(TESTS)}"
    )

    print(
        "NONEMPTY GENERATION: "
        f"{generation_nonempty}/"
        f"{len(TESTS)}"
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "END-TO-END TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()