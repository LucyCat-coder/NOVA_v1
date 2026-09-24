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


from rag import Embedder, Retriever  # noqa: E402


INDEX_PATH = (
    PROJECT_ROOT
    / "rag"
    / "trusted_index"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "trusted_rag_test.txt"
)


CORE_TESTS = [
    (
        "core_tcp_udp",
        "TCP надёжный, а UDP нет — в чём между ними реальная разница?",
    ),
    (
        "core_docker_vm",
        "Контейнер Docker и обычная VM используют систему одинаково или нет?",
    ),
    (
        "core_python_mutable_default",
        "Почему список в default-аргументе Python сохраняет изменения между вызовами?",
    ),
    (
        "core_git_merge_rebase",
        "Merge сохраняет историю, а rebase переписывает? Объясни различие.",
    ),
    (
        "core_pandas_missing",
        "Чем в pandas лучше заменить NaN?",
    ),
    (
        "core_postgresql",
        "Postgres — это язык или СУБД?",
    ),
    (
        "core_database_index",
        "Зачем базе данных индекс и какой у него минус?",
    ),
    (
        "core_rest_api",
        "REST — это просто HTTP с JSON?",
    ),
    (
        "core_http",
        "Как устроен обмен запросом и ответом в HTTP?",
    ),
    (
        "core_sql",
        "Для чего используется SQL?",
    ),
    (
        "core_python",
        "Python — что это за язык?",
    ),
    (
        "core_docker",
        "Для чего используют Docker?",
    ),
]


TECHNICAL_PROBES = [
    "Когда в Python использовать is, а когда ==?",
    "Что такое замыкание в Python?",
    "Чем docker stop отличается от docker kill?",
    "Чем git fetch отличается от git pull?",
    "Чем HTTP 401 отличается от 403 и 404?",
    "Что такое Docker volume?",
    "Что такое NAT?",
    "Что такое SQL view?",
    "Что такое API key?",
    "Что означает merge conflict?",
    "Чем Content-Type отличается от Accept?",
    "Чем COUNT(*) отличается от COUNT(column)?",
]


NEGATIVE_TESTS = [
    "Как приготовить борщ?",
    "Сколько будет 17 умножить на 23?",
    "Почему небо голубое?",
    "Как заменить колесо на автомобиле?",
    "Напиши поздравление с днем рождения.",
    "Как работает квантовая запутанность?",
    "Какая столица Австралии?",
    "Как настроить nginx reverse proxy с двумя upstream?",
    "Почему CUDA kernel получает out of memory?",
    "Как реализовать бинарное дерево поиска на C++?",
    "Что такое Kubernetes StatefulSet?",
    "Как работает OAuth 2.0 authorization code flow?",
]


def get_results(
    retriever: Retriever,
    query: str,
):
    return retriever.retrieve(
        query=query,
        top_k=3,
        min_score=None,
        use_reranker=False,
    )


def append_results(
    lines: list[str],
    results,
):
    for rank, (
        answer,
        score,
        metadata,
    ) in enumerate(
        results,
        start=1,
    ):
        lines.append(
            f"#{rank} "
            f"score={score:.6f}"
        )

        lines.append(
            f"id={metadata.get('id')}"
        )

        lines.append(
            "source_type="
            f"{metadata.get('source_type')}"
        )

        lines.append(
            "question:"
        )

        lines.append(
            str(
                metadata.get(
                    "question",
                    "",
                )
            )
        )

        lines.append("")


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — TRUSTED RAG TEST"
    )
    print("=" * 100)
    print()

    embedder = Embedder(
        device="cuda",
        pooling_mode="masked_mean",
    )

    retriever = Retriever(
        index_path=INDEX_PATH,
        embedder=embedder,
    )

    print(
        f"Trusted records: "
        f"{len(retriever):,}"
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA TRUSTED RAG TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        f"Records: {len(retriever)}"
    )

    lines.append("")

    core_correct = 0
    core_scores = []
    negative_scores = []

    started = time.time()

    # --------------------------------------------------------
    # CORE PARAPHRASES
    # --------------------------------------------------------

    lines.append(
        "=" * 100
    )

    lines.append(
        "CORE PARAPHRASES"
    )

    lines.append(
        "=" * 100
    )

    for number, (
        expected_id,
        query,
    ) in enumerate(
        CORE_TESTS,
        start=1,
    ):
        results = get_results(
            retriever,
            query,
        )

        top1_id = (
            results[0][2].get("id")
            if results
            else None
        )

        top1_score = (
            float(results[0][1])
            if results
            else None
        )

        if top1_id == expected_id:
            core_correct += 1

        if top1_score is not None:
            core_scores.append(
                top1_score
            )

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"CORE {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"EXPECTED: {expected_id}"
        )

        lines.append(
            f"TOP1: {top1_id}"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if top1_id
                == expected_id
                else "FAIL"
            )
        )

        append_results(
            lines,
            results,
        )

    # --------------------------------------------------------
    # TECHNICAL PROBES
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "TECHNICAL PROBES"
    )

    lines.append(
        "=" * 100
    )

    for number, query in enumerate(
        TECHNICAL_PROBES,
        start=1,
    ):
        results = get_results(
            retriever,
            query,
        )

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"TECHNICAL {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        append_results(
            lines,
            results,
        )

    # --------------------------------------------------------
    # OUT OF DOMAIN
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "OUT OF DOMAIN"
    )

    lines.append(
        "=" * 100
    )

    for number, query in enumerate(
        NEGATIVE_TESTS,
        start=1,
    ):
        results = get_results(
            retriever,
            query,
        )

        if results:
            negative_scores.append(
                float(
                    results[0][1]
                )
            )

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"NEGATIVE {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        append_results(
            lines,
            results,
        )

    total_time = (
        time.time()
        - started
    )

    lines.append("")
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
        "CORE accuracy: "
        f"{core_correct}/"
        f"{len(CORE_TESTS)}"
    )

    if core_scores:
        lines.append(
            "CORE TOP1 score: "
            f"min={min(core_scores):.6f}, "
            f"max={max(core_scores):.6f}"
        )

    if negative_scores:
        lines.append(
            "OOD TOP1 score: "
            f"min={min(negative_scores):.6f}, "
            f"max={max(negative_scores):.6f}"
        )

    lines.append(
        f"Total time: "
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
        "CORE accuracy: "
        f"{core_correct}/"
        f"{len(CORE_TESTS)}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "TRUSTED RAG TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()