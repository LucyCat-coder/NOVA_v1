from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
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


from rag import Embedder, VectorStore  # noqa: E402


BROAD_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2.jsonl"
)

TECHNICAL_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "technical_index.jsonl"
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "rag"
    / "question_router"
)

TEMP_BASE = (
    PROJECT_ROOT
    / "rag"
    / "question_router_building"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "question_router_build_report.txt"
)


MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

BATCH_SIZE = 64


def sha256_file(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def normalize_question(
    text: str,
) -> str:
    return " ".join(
        str(text)
        .strip()
        .lower()
        .replace("ё", "е")
        .split()
    )


def load_broad_questions():
    """
    knowledge_index_v2.jsonl содержит 77 900 chunks.

    Нам здесь нужен НЕ каждый chunk, а только
    уникальный исходный вопрос.

    В результате должно получиться примерно
    15 000 broad questions.
    """

    result = []
    seen = set()

    with BROAD_JSONL.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            meta = record.get(
                "meta",
                {},
            )

            if not isinstance(
                meta,
                dict,
            ):
                continue

            question = str(
                meta.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                continue

            q_idx = meta.get(
                "q_idx",
                None,
            )

            if q_idx is not None:
                key = (
                    "broad",
                    str(q_idx),
                )
            else:
                key = (
                    "broad-question",
                    normalize_question(
                        question
                    ),
                )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                {
                    "question": question,
                    "meta": {
                        "source": (
                            "nova_knowledge"
                        ),
                        "source_type": (
                            "broad"
                        ),
                        "q_idx": q_idx,
                        "original_line": (
                            line_number
                        ),
                    },
                }
            )

    return result


def load_technical_questions():
    """
    technical_index.jsonl уже содержит
    одну запись на один technical Q/A.
    """

    result = []

    with TECHNICAL_JSONL.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            meta = record.get(
                "meta",
                {},
            )

            if not isinstance(
                meta,
                dict,
            ):
                continue

            question = str(
                meta.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                continue

            result.append(
                {
                    "question": question,
                    "meta": {
                        "source": (
                            "nova_technical_v1_1"
                        ),
                        "source_type": (
                            "technical"
                        ),
                        "line": meta.get(
                            "line",
                            line_number,
                        ),
                        "original_line": (
                            line_number
                        ),
                    },
                }
            )

    return result


def remove_files(
    base: Path,
):
    for suffix in [
        ".faiss",
        ".jsonl",
    ]:
        path = Path(
            str(base)
            + suffix
        )

        if path.exists():
            path.unlink()


def output_exists() -> bool:
    return any(
        Path(
            str(OUTPUT_BASE)
            + suffix
        ).exists()
        for suffix in [
            ".faiss",
            ".jsonl",
        ]
    )


def count_jsonl(
    path: Path,
) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build NOVA question-only "
            "router index."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить перестроить "
            "существующий question_router."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "NOVA — BUILD QUESTION ROUTER"
    )
    print("=" * 80)
    print()

    print(
        f"BROAD:     {BROAD_JSONL}"
    )

    print(
        f"TECHNICAL: {TECHNICAL_JSONL}"
    )

    print(
        f"OUTPUT:    "
        f"{OUTPUT_BASE}.{{faiss,jsonl}}"
    )

    print()

    print(
        "Embedding contains ONLY "
        "the source question."
    )

    print(
        "Answers/chunks are NOT embedded."
    )

    print(
        "Pooling: masked_mean"
    )

    print()

    if not BROAD_JSONL.exists():
        raise FileNotFoundError(
            BROAD_JSONL
        )

    if not TECHNICAL_JSONL.exists():
        raise FileNotFoundError(
            TECHNICAL_JSONL
        )

    if output_exists():
        if not args.force:
            raise RuntimeError(
                "question_router уже существует.\n"
                "Ничего не перезаписано.\n"
                "Для rebuild используй --force."
            )

        print(
            "Removing existing "
            "question_router..."
        )

        remove_files(
            OUTPUT_BASE
        )

    remove_files(
        TEMP_BASE
    )

    started_total = time.time()

    print(
        "Calculating source hashes..."
    )

    broad_hash = sha256_file(
        BROAD_JSONL
    )

    technical_hash = sha256_file(
        TECHNICAL_JSONL
    )

    print(
        f"BROAD SHA-256:     "
        f"{broad_hash}"
    )

    print(
        f"TECHNICAL SHA-256: "
        f"{technical_hash}"
    )

    print()
    print(
        "Reading unique questions..."
    )

    broad_records = (
        load_broad_questions()
    )

    technical_records = (
        load_technical_questions()
    )

    records = (
        broad_records
        + technical_records
    )

    print(
        f"BROAD questions:     "
        f"{len(broad_records):,}"
    )

    print(
        f"TECHNICAL questions: "
        f"{len(technical_records):,}"
    )

    print(
        f"TOTAL:               "
        f"{len(records):,}"
    )

    if not records:
        raise RuntimeError(
            "Question router dataset пуст."
        )

    source_counts = Counter(
        item[
            "meta"
        ][
            "source_type"
        ]
        for item in records
    )

    print()
    print(
        "Loading embedder..."
    )

    embedder = Embedder(
        model_name=MODEL_NAME,
        device="cuda",
        pooling_mode="masked_mean",
    )

    store = VectorStore(
        dim=embedder.dim
    )

    print()
    print(
        "Building question embeddings..."
    )

    started_embeddings = (
        time.time()
    )

    total = len(
        records
    )

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = records[
            start:end
        ]

        questions = [
            item[
                "question"
            ]
            for item in batch
        ]

        metadata = [
            item[
                "meta"
            ]
            for item in batch
        ]

        embeddings = (
            embedder.encode_passages(
                questions,
                batch_size=BATCH_SIZE,
            )
        )

        # Здесь text = исходный вопрос.
        #
        # Это диагностический question-level
        # retrieval, а не обычный answer RAG.
        store.add(
            texts=questions,
            embeddings=embeddings,
            metadata=metadata,
        )

        elapsed = (
            time.time()
            - started_embeddings
        )

        rate = (
            end / elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"\rProcessed "
            f"{end:,}/{total:,} "
            f"({end / total * 100:5.1f}%) "
            f"| {rate:,.1f} questions/s",
            end="",
            flush=True,
        )

    print()

    embedding_time = (
        time.time()
        - started_embeddings
    )

    print()
    print(
        "Saving temporary index..."
    )

    store.save(
        TEMP_BASE
    )

    temp_faiss = Path(
        str(TEMP_BASE)
        + ".faiss"
    )

    temp_jsonl = Path(
        str(TEMP_BASE)
        + ".jsonl"
    )

    print()
    print(
        "Verifying temporary index..."
    )

    index = faiss.read_index(
        str(temp_faiss)
    )

    jsonl_count = count_jsonl(
        temp_jsonl
    )

    print(
        f"FAISS dimension: {index.d}"
    )

    print(
        f"FAISS ntotal:    "
        f"{index.ntotal:,}"
    )

    print(
        f"JSONL records:   "
        f"{jsonl_count:,}"
    )

    if (
        index.d
        != embedder.dim
    ):
        raise RuntimeError(
            "Неверная размерность "
            "question router."
        )

    if (
        index.ntotal
        != total
    ):
        raise RuntimeError(
            "FAISS ntotal не совпадает "
            "с количеством вопросов."
        )

    if (
        jsonl_count
        != total
    ):
        raise RuntimeError(
            "JSONL count не совпадает "
            "с количеством вопросов."
        )

    print(
        "Temporary index: OK"
    )

    print()
    print(
        "Moving temporary index "
        "to final paths..."
    )

    final_faiss = Path(
        str(OUTPUT_BASE)
        + ".faiss"
    )

    final_jsonl = Path(
        str(OUTPUT_BASE)
        + ".jsonl"
    )

    temp_faiss.replace(
        final_faiss
    )

    temp_jsonl.replace(
        final_jsonl
    )

    total_time = (
        time.time()
        - started_total
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = [
        "NOVA QUESTION ROUTER BUILD REPORT",
        "=" * 80,
        "",
        f"BROAD source: {BROAD_JSONL}",
        f"BROAD SHA-256: {broad_hash}",
        "",
        (
            "TECHNICAL source: "
            f"{TECHNICAL_JSONL}"
        ),
        (
            "TECHNICAL SHA-256: "
            f"{technical_hash}"
        ),
        "",
        f"Model: {MODEL_NAME}",
        "Pooling: masked_mean",
        "Embedding: source question ONLY",
        "",
        (
            "BROAD questions: "
            f"{source_counts['broad']}"
        ),
        (
            "TECHNICAL questions: "
            f"{source_counts['technical']}"
        ),
        (
            "TOTAL questions: "
            f"{total}"
        ),
        "",
        (
            "Embedding dimension: "
            f"{index.d}"
        ),
        "",
        f"FAISS: {final_faiss}",
        f"JSONL: {final_jsonl}",
        "",
        (
            "Embedding time: "
            f"{embedding_time:.1f}s"
        ),
        (
            "Total time: "
            f"{total_time:.1f}s"
        ),
    ]

    REPORT_PATH.write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "QUESTION ROUTER BUILD COMPLETE"
    )
    print("=" * 80)

    print(
        f"FAISS: {final_faiss}"
    )

    print(
        f"JSONL: {final_jsonl}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Total time: {total_time:.1f}s"
    )

    print()
    print(
        "No existing RAG index "
        "was modified."
    )


if __name__ == "__main__":
    main()