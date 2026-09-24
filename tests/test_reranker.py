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


def compact_text(
    text: str,
    limit: int = 450,
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
        text = (
            text[:limit]
            + "..."
        )

    return text


def print_results(
    title: str,
    results,
):
    print()
    print(title)
    print("-" * 100)

    if not results:
        print(
            "(no results)"
        )
        return

    for rank, (
        text,
        score,
        metadata,
    ) in enumerate(
        results,
        start=1,
    ):
        question = metadata.get(
            "question",
            "(no question metadata)",
        )

        q_idx = metadata.get(
            "q_idx",
            "?",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "?",
        )

        print()
        print(
            f"#{rank} "
            f"score={score:.6f}"
        )

        print(
            f"source question: "
            f"{question}"
        )

        print(
            f"q_idx={q_idx}, "
            f"chunk_id={chunk_id}"
        )

        print(
            compact_text(
                text
            )
        )


def main():
    print()
    print("=" * 100)
    print(
        "NOVA RAG — FAISS vs RERANKER"
    )
    print("=" * 100)

    print()
    print(
        "FAISS comparison:"
    )

    print(
        "  show top 5"
    )

    print(
        "Reranker comparison:"
    )

    print(
        "  retrieve 15 FAISS candidates"
    )

    print(
        "  rerank them"
    )

    print(
        "  show top 5"
    )

    print()
    print(
        "min_score = NONE"
    )

    print(
        "Generation = OFF"
    )

    print()

    started = time.time()

    retriever = Retriever(
        index_path=INDEX_PATH,
        device="cuda",
    )

    print()
    print(
        f"Retriever ready in "
        f"{time.time() - started:.2f}s"
    )

    print(
        f"Records: {len(retriever)}"
    )

    print()

    all_started = time.time()

    for number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print()
        print("=" * 100)

        print(
            f"[{number}/{len(QUESTIONS)}]"
        )

        print(
            f"QUERY: {question}"
        )

        print("=" * 100)

        # ----------------------------------------------------
        # FAISS ONLY
        # ----------------------------------------------------

        started = time.time()

        faiss_results = (
            retriever.retrieve(
                query=question,
                top_k=5,
                min_score=None,
                use_reranker=False,
            )
        )

        faiss_time = (
            time.time()
            - started
        )

        print_results(
            "FAISS TOP 5",
            faiss_results,
        )

        print()
        print(
            f"FAISS time: "
            f"{faiss_time:.3f}s"
        )

        # ----------------------------------------------------
        # FAISS + RERANKER
        # ----------------------------------------------------

        started = time.time()

        reranked_results = (
            retriever.retrieve(
                query=question,

                # Retriever автоматически
                # возьмёт top_k * 3,
                # то есть 15 кандидатов.
                top_k=5,

                min_score=None,

                use_reranker=True,

                rerank_top_k=5,
            )
        )

        rerank_time = (
            time.time()
            - started
        )

        print_results(
            "RERANKED TOP 5",
            reranked_results,
        )

        print()
        print(
            f"Reranker time: "
            f"{rerank_time:.3f}s"
        )

    print()
    print("=" * 100)

    print(
        "RERANKER TEST FINISHED"
    )

    print(
        "Total time: "
        f"{time.time() - all_started:.1f}s"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()