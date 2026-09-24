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


from rag import (  # noqa: E402
    Embedder,
    Retriever,
)


OLD_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index"
)

NEW_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rag_v1_vs_v2.txt"
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
]


def compact(
    text: str,
    limit: int = 500,
) -> str:
    text = (
        text
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


def unload(
    retriever,
):
    if retriever is not None:
        try:
            retriever.embedder.model.to(
                "cpu"
            )
        except Exception:
            pass

        del retriever

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_results(
    *,
    index_path: Path,
    pooling_mode: str,
):
    print()
    print("=" * 100)

    print(
        f"Loading {index_path.name}"
    )

    print(
        f"pooling={pooling_mode}"
    )

    print("=" * 100)

    embedder = Embedder(
        device="cuda",
        pooling_mode=pooling_mode,
    )

    retriever = Retriever(
        index_path=index_path,
        embedder=embedder,
    )

    all_results = []

    try:
        for number, question in enumerate(
            QUESTIONS,
            start=1,
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

            print(
                f"[{number}/{len(QUESTIONS)}] "
                f"{elapsed:.3f}s "
                f"{question}"
            )

            all_results.append(
                {
                    "question": question,
                    "elapsed": elapsed,
                    "results": results,
                }
            )

    finally:
        unload(
            retriever
        )

    return all_results


def write_result_set(
    lines: list[str],
    title: str,
    result_set,
):
    lines.append(title)
    lines.append(
        "-" * 100
    )

    for rank, (
        text,
        score,
        metadata,
    ) in enumerate(
        result_set,
        start=1,
    ):
        source_question = (
            metadata.get(
                "question",
                "(no question)",
            )
        )

        q_idx = metadata.get(
            "q_idx",
            "?",
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
            "source question: "
            f"{source_question}"
        )

        lines.append(
            f"q_idx={q_idx}, "
            f"chunk_id={chunk_id}"
        )

        lines.append(
            compact(text)
        )

        lines.append("")


def main():
    print()
    print("=" * 100)
    print("NOVA — RAG INDEX V1 vs V2")
    print("=" * 100)

    if not Path(
        str(OLD_INDEX)
        + ".faiss"
    ).exists():
        raise FileNotFoundError(
            "Old FAISS index отсутствует."
        )

    if not Path(
        str(NEW_INDEX)
        + ".faiss"
    ).exists():
        raise FileNotFoundError(
            "New RAG v2 index отсутствует. "
            "Сначала запусти "
            "scripts/build_rag_index.py"
        )

    started_all = time.time()

    old_results = collect_results(
        index_path=OLD_INDEX,
        pooling_mode="legacy_mean",
    )

    new_results = collect_results(
        index_path=NEW_INDEX,
        pooling_mode="masked_mean",
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA RAG INDEX COMPARISON"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "V1:"
    )

    lines.append(
        "  passage embedding = answer chunk"
    )

    lines.append(
        "  pooling = legacy_mean"
    )

    lines.append("")

    lines.append(
        "V2:"
    )

    lines.append(
        "  passage embedding = "
        "question + answer chunk"
    )

    lines.append(
        "  pooling = masked_mean"
    )

    lines.append("")

    lines.append(
        "No reranker."
    )

    lines.append(
        "No min_score."
    )

    lines.append(
        "Top 5."
    )

    lines.append("")

    for index, question in enumerate(
        QUESTIONS
    ):
        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUESTION {index + 1}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUERY: {question}"
        )

        lines.append("")

        write_result_set(
            lines,
            "V1 — OLD INDEX",
            old_results[
                index
            ]["results"],
        )

        write_result_set(
            lines,
            "V2 — NEW INDEX",
            new_results[
                index
            ]["results"],
        )

    total_time = (
        time.time()
        - started_all
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"Total time: {total_time:.1f}s"
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
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()