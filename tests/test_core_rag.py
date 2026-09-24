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
    / "core_index"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "core_rag_test.txt"
)


TESTS = [
    (
        "core_tcp_udp",
        "Что такое TCP и чем он отличается от UDP?",
    ),

    (
        "core_docker_vm",
        "Чем Docker-контейнер отличается от виртуальной машины?",
    ),

    (
        "core_python_mutable_default",
        "Почему нельзя использовать изменяемый объект как значение аргумента по умолчанию в Python?",
    ),

    (
        "core_git_merge_rebase",
        "В чём разница между git merge и git rebase?",
    ),

    (
        "core_pandas_missing",
        "Как заполнить пропущенные значения в pandas DataFrame?",
    ),

    (
        "core_postgresql",
        "Что такое PostgreSQL?",
    ),

    (
        "core_database_index",
        "Как работает индекс в базе данных?",
    ),

    (
        "core_rest_api",
        "Что такое REST API?",
    ),

    (
        "core_http",
        "Что такое HTTP?",
    ),

    (
        "core_sql",
        "Что такое SQL?",
    ),

    (
        "core_python",
        "Что такое Python?",
    ),

    (
        "core_docker",
        "Что такое Docker?",
    ),
]


def index_exists() -> bool:
    return (
        Path(
            str(INDEX_PATH)
            + ".faiss"
        ).exists()
        and Path(
            str(INDEX_PATH)
            + ".jsonl"
        ).exists()
    )


def compact(
    text: str,
    limit: int = 500,
) -> str:
    text = (
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )

    while "  " in text:
        text = text.replace(
            "  ",
            " ",
        )

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
        "NOVA — CORE RAG TEST"
    )
    print("=" * 100)
    print()

    if not index_exists():
        raise FileNotFoundError(
            "core_index отсутствует. "
            "Сначала запусти "
            "scripts/build_core_rag_index.py"
        )

    print(
        "Loading embedder..."
    )

    embedder = Embedder(
        device="cuda",
        pooling_mode="masked_mean",
    )

    print()
    print(
        "Loading CORE index..."
    )

    retriever = Retriever(
        index_path=INDEX_PATH,
        embedder=embedder,
    )

    print()
    print(
        f"CORE records: "
        f"{len(retriever):,}"
    )

    print()

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA CORE RAG TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        f"CORE records: "
        f"{len(retriever)}"
    )

    lines.append(
        "Top 3 per query."
    )

    lines.append(
        "Expected result must be TOP 1."
    )

    lines.append("")

    passed = 0

    started_all = time.time()

    for number, (
        expected_id,
        query,
    ) in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{number}/{len(TESTS)}] "
            f"{query}"
        )

        started = time.time()

        results = retriever.retrieve(
            query=query,
            top_k=3,
            min_score=None,
            use_reranker=False,
        )

        elapsed = (
            time.time()
            - started
        )

        top1_id = None

        if results:
            top1_id = (
                results[0][2]
                .get(
                    "id"
                )
            )

        ok = (
            top1_id
            == expected_id
        )

        if ok:
            passed += 1

        lines.append(
            "=" * 100
        )

        lines.append(
            f"TEST {number}"
        )

        lines.append(
            "=" * 100
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
                if ok
                else "FAIL"
            )
        )

        lines.append(
            f"time={elapsed:.4f}s"
        )

        lines.append("")

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
                "id="
                f"{metadata.get('id')}"
            )

            lines.append(
                "question:"
            )

            lines.append(
                str(
                    metadata.get(
                        "question",
                        ""
                    )
                )
            )

            lines.append(
                "answer:"
            )

            lines.append(
                compact(
                    answer
                )
            )

            lines.append("")

    total_time = (
        time.time()
        - started_all
    )

    total = len(
        TESTS
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
        f"PASS: {passed}/{total}"
    )

    lines.append(
        f"FAIL: {total - passed}/{total}"
    )

    lines.append(
        "ACCURACY: "
        f"{passed / total * 100:.1f}%"
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
        f"PASS: {passed}/{total}"
    )

    print(
        "ACCURACY: "
        f"{passed / total * 100:.1f}%"
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "CORE RAG TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()