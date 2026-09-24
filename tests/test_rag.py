from __future__ import annotations

import sys
import time
from pathlib import Path

import faiss


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


from config import RUNTIME_CONFIG  # noqa: E402
from rag import Retriever  # noqa: E402


INDEX_PATH = (
    RUNTIME_CONFIG[
        "rag_index"
    ]
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


def inspect_index():
    faiss_path = Path(
        str(INDEX_PATH)
        + ".faiss"
    )

    jsonl_path = Path(
        str(INDEX_PATH)
        + ".jsonl"
    )

    print("=" * 80)
    print("RAG INDEX STRUCTURE")
    print("=" * 80)

    print(
        f"FAISS: {faiss_path}"
    )

    print(
        f"JSONL: {jsonl_path}"
    )

    if not faiss_path.exists():
        raise FileNotFoundError(
            faiss_path
        )

    if not jsonl_path.exists():
        raise FileNotFoundError(
            jsonl_path
        )

    index = faiss.read_index(
        str(faiss_path)
    )

    print(
        f"dimension: {index.d}"
    )

    print(
        f"ntotal:    {index.ntotal}"
    )

    jsonl_count = 0

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                jsonl_count += 1

    print(
        f"jsonl:     {jsonl_count}"
    )

    if (
        index.ntotal
        != jsonl_count
    ):
        raise RuntimeError(
            "FAISS и JSONL "
            "не совпадают по размеру."
        )

    print(
        "Index structure: OK"
    )

    print()


def main():
    inspect_index()

    print("=" * 80)
    print("LOADING RETRIEVER")
    print("=" * 80)
    print()
    print(
        "При первом запуске Hugging Face "
        "может скачать multilingual-e5-small."
    )
    print()

    started = time.time()

    retriever = Retriever(
        index_path=INDEX_PATH,
        device="cuda",
    )

    print()
    print(
        f"Retriever loaded in "
        f"{time.time() - started:.1f}s"
    )

    print(
        f"records: {len(retriever)}"
    )

    print(
        f"embedding dim: "
        f"{retriever.embedder.dim}"
    )

    print()
    print("=" * 80)
    print("RETRIEVAL TEST")
    print("=" * 80)

    for number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print()
        print(
            "=" * 80
        )

        print(
            f"[{number}/{len(QUESTIONS)}]"
        )

        print(
            f"QUERY: {question}"
        )

        print(
            "-" * 80
        )

        started = time.time()

        # ВАЖНО:
        # никакого min_score сейчас нет.
        #
        # Сначала смотрим реальные
        # score существующего индекса.
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

        if not results:
            print(
                "NO RESULTS"
            )

            continue

        for rank, (
            text,
            score,
            metadata,
        ) in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"#{rank} "
                f"score={score:.6f}"
            )

            print(
                f"meta={metadata}"
            )

            print()

            clean_text = (
                text
                .replace(
                    "\r",
                    " ",
                )
                .strip()
            )

            if (
                len(clean_text)
                > 800
            ):
                clean_text = (
                    clean_text[:800]
                    + "..."
                )

            print(
                clean_text
            )

        print()
        print(
            f"Retrieval time: "
            f"{elapsed:.3f}s"
        )

    print()
    print("=" * 80)
    print("RAG RETRIEVAL TEST FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()