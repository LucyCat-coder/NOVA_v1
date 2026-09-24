from __future__ import annotations

import sys
import time
from collections import Counter
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
    / "question_router"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "question_router_test.txt"
)


QUESTIONS = [
    "Что такое TCP и чем он отличается от UDP?",

    "Чем Docker-контейнер отличается от виртуальной машины?",

    "Почему нельзя использовать изменяемый объект как значение аргумента по умолчанию в Python?",

    "В чём разница между git merge и git rebase?",

    "Как заполнить пропущенные значения в pandas DataFrame?",

    "Что такое PostgreSQL?",

    "Как работает индекс в базе данных?",

    "Что такое REST API?",

    "Что такое HTTP?",

    "Что такое SQL?",

    "Что такое Python?",

    "Что такое Docker?",
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


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — QUESTION ROUTER TEST"
    )
    print("=" * 100)
    print()

    if not index_exists():
        raise FileNotFoundError(
            "question_router index отсутствует. "
            "Сначала запусти "
            "scripts/build_question_index.py"
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
        "Loading question router..."
    )

    retriever = Retriever(
        index_path=INDEX_PATH,
        embedder=embedder,
    )

    print()
    print(
        f"Router records: "
        f"{len(retriever):,}"
    )

    print()
    print(
        "No reranker."
    )

    print(
        "No threshold."
    )

    print(
        "Top 10 source questions."
    )

    print()

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA QUESTION ROUTER TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Embedding:"
    )

    lines.append(
        "  intfloat/multilingual-e5-small"
    )

    lines.append(
        "  masked_mean"
    )

    lines.append(
        "  indexed text = source question only"
    )

    lines.append("")

    lines.append(
        f"Router records: "
        f"{len(retriever)}"
    )

    lines.append(
        "No reranker."
    )

    lines.append(
        "No threshold."
    )

    lines.append(
        "Top 10."
    )

    lines.append("")

    started_all = time.time()

    for number, query in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            f"[{number}/{len(QUESTIONS)}] "
            f"{query}"
        )

        started = time.time()

        results = retriever.retrieve(
            query=query,
            top_k=10,
            min_score=None,
            use_reranker=False,
        )

        elapsed = (
            time.time()
            - started
        )

        source_counter = Counter(
            metadata.get(
                "source_type",
                "?",
            )
            for _, _, metadata
            in results
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUESTION {number}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"time={elapsed:.4f}s"
        )

        lines.append(
            "top10 source counts: "
            f"{dict(source_counter)}"
        )

        lines.append("")

        for rank, (
            source_question,
            score,
            metadata,
        ) in enumerate(
            results,
            start=1,
        ):
            source_type = (
                metadata.get(
                    "source_type",
                    "?",
                )
            )

            source = (
                metadata.get(
                    "source",
                    "?",
                )
            )

            q_idx = metadata.get(
                "q_idx",
                None,
            )

            line = metadata.get(
                "line",
                None,
            )

            lines.append(
                f"#{rank} "
                f"score={score:.6f} "
                f"type={source_type}"
            )

            lines.append(
                f"source={source}"
            )

            if q_idx is not None:
                lines.append(
                    f"q_idx={q_idx}"
                )

            if line is not None:
                lines.append(
                    f"line={line}"
                )

            lines.append(
                "SOURCE QUESTION:"
            )

            lines.append(
                str(
                    source_question
                ).strip()
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
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    print()
    print("=" * 100)
    print(
        "QUESTION ROUTER TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()