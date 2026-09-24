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


BROAD_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2"
)

TECHNICAL_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "technical_index"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "broad_vs_technical_rag.txt"
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


def compact(
    text: str,
    limit: int = 650,
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


def index_exists(
    base: Path,
) -> bool:
    return (
        Path(
            str(base)
            + ".faiss"
        ).exists()
        and Path(
            str(base)
            + ".jsonl"
        ).exists()
    )


def retrieve(
    retriever: Retriever,
    question: str,
):
    started = time.time()

    results = retriever.retrieve(
        query=question,
        top_k=5,
        min_score=None,
        use_reranker=False,
    )

    elapsed = (
        time.time()
        - started
    )

    return (
        results,
        elapsed,
    )


def append_results(
    lines: list[str],
    title: str,
    results,
    elapsed: float,
):
    lines.append(
        title
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        f"time={elapsed:.4f}s"
    )

    lines.append("")

    if not results:
        lines.append(
            "(no results)"
        )

        lines.append("")

        return

    for rank, (
        text,
        score,
        metadata,
    ) in enumerate(
        results,
        start=1,
    ):
        source_question = (
            metadata.get(
                "question",
                "(no source question)",
            )
        )

        source = metadata.get(
            "source",
            "?",
        )

        q_idx = metadata.get(
            "q_idx",
            None,
        )

        line = metadata.get(
            "line",
            None,
        )

        chunk_id = metadata.get(
            "chunk_id",
            "?",
        )

        lines.append(
            f"#{rank} "
            f"score={score:.6f}"
        )

        lines.append(
            f"source={source}"
        )

        if q_idx is not None:
            lines.append(
                f"q_idx={q_idx}, "
                f"chunk_id={chunk_id}"
            )

        elif line is not None:
            lines.append(
                f"line={line}, "
                f"chunk_id={chunk_id}"
            )

        lines.append(
            "source question:"
        )

        lines.append(
            compact(
                source_question,
                500,
            )
        )

        lines.append(
            "answer/chunk:"
        )

        lines.append(
            compact(
                text,
                650,
            )
        )

        lines.append("")


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — BROAD V2 vs TECHNICAL RAG"
    )
    print("=" * 100)
    print()

    if not index_exists(
        BROAD_INDEX
    ):
        raise FileNotFoundError(
            "Broad V2 index отсутствует."
        )

    if not index_exists(
        TECHNICAL_INDEX
    ):
        raise FileNotFoundError(
            "Technical index отсутствует. "
            "Сначала запусти "
            "scripts/build_technical_rag_index.py"
        )

    print(
        "Loading one shared embedder..."
    )

    embedder = Embedder(
        device="cuda",
        pooling_mode="masked_mean",
    )

    print()
    print(
        "Loading BROAD V2..."
    )

    broad = Retriever(
        index_path=BROAD_INDEX,
        embedder=embedder,
    )

    print()
    print(
        f"BROAD records: "
        f"{len(broad):,}"
    )

    print()
    print(
        "Loading TECHNICAL..."
    )

    technical = Retriever(
        index_path=TECHNICAL_INDEX,
        embedder=embedder,
    )

    print()
    print(
        f"TECHNICAL records: "
        f"{len(technical):,}"
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA — BROAD V2 vs TECHNICAL RAG"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Shared embedder:"
    )

    lines.append(
        "  intfloat/multilingual-e5-small"
    )

    lines.append(
        "  pooling=masked_mean"
    )

    lines.append("")

    lines.append(
        "BROAD V2:"
    )

    lines.append(
        f"  index={BROAD_INDEX}"
    )

    lines.append(
        f"  records={len(broad)}"
    )

    lines.append("")

    lines.append(
        "TECHNICAL:"
    )

    lines.append(
        f"  index={TECHNICAL_INDEX}"
    )

    lines.append(
        f"  records={len(technical)}"
    )

    lines.append("")

    lines.append(
        "No reranker."
    )

    lines.append(
        "No min_score."
    )

    lines.append(
        "Top 5 from each index."
    )

    lines.append("")

    started_all = time.time()

    for number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            f"[{number}/{len(QUESTIONS)}] "
            f"{question}"
        )

        broad_results, broad_time = (
            retrieve(
                broad,
                question,
            )
        )

        (
            technical_results,
            technical_time,
        ) = retrieve(
            technical,
            question,
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
            f"QUERY: {question}"
        )

        lines.append("")

        append_results(
            lines,
            "BROAD V2",
            broad_results,
            broad_time,
        )

        append_results(
            lines,
            "TECHNICAL",
            technical_results,
            technical_time,
        )

    total_time = (
        time.time()
        - started_all
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"Total comparison time: "
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
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Comparison time: "
        f"{total_time:.2f}s"
    )

    print()
    print("=" * 100)
    print(
        "COMPARISON FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()