from __future__ import annotations

import argparse
import hashlib
import json
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


from rag import Embedder, VectorStore  # noqa: E402


CORE_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "core_index.jsonl"
)

TECHNICAL_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "technical_index.jsonl"
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "rag"
    / "trusted_index"
)

TEMP_BASE = (
    PROJECT_ROOT
    / "rag"
    / "trusted_index_building"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "trusted_rag_build_report.txt"
)


MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

BATCH_SIZE = 64


def sha256_file(
    path: Path,
    chunk_size: int = 4 * 1024 * 1024,
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


def read_index_jsonl(
    path: Path,
):
    records = []

    with path.open(
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

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(
                    f"{path.name}, строка "
                    f"{line_number}: "
                    "ожидался JSON object."
                )

            text = str(
                record.get(
                    "text",
                    "",
                )
            ).strip()

            meta = record.get(
                "meta",
                {},
            )

            if not isinstance(
                meta,
                dict,
            ):
                raise RuntimeError(
                    f"{path.name}, строка "
                    f"{line_number}: "
                    "meta должен быть object."
                )

            if not text:
                raise RuntimeError(
                    f"{path.name}, строка "
                    f"{line_number}: "
                    "нет text."
                )

            records.append(
                (
                    text,
                    meta,
                    line_number,
                )
            )

    return records


def load_trusted_records():
    result = []
    seen_questions = set()

    skipped_duplicates = 0

    # --------------------------------------------------------
    # CORE FIRST
    #
    # CORE имеет приоритет при точном совпадении
    # normalized question.
    # --------------------------------------------------------

    core_records = read_index_jsonl(
        CORE_JSONL
    )

    for answer, meta, line_number in (
        core_records
    ):
        question = str(
            meta.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            raise RuntimeError(
                "CORE record без question: "
                f"строка {line_number}"
            )

        aliases = meta.get(
            "aliases",
            [],
        )

        if not isinstance(
            aliases,
            list,
        ):
            aliases = []

        aliases = [
            str(alias).strip()
            for alias in aliases
            if str(alias).strip()
        ]

        normalized = normalize_question(
            question
        )

        if normalized in seen_questions:
            skipped_duplicates += 1
            continue

        seen_questions.add(
            normalized
        )

        parts = [
            f"Вопрос: {question}"
        ]

        if aliases:
            parts.append(
                "Альтернативные формулировки:"
            )

            for alias in aliases:
                parts.append(
                    f"- {alias}"
                )

        embedding_text = "\n".join(
            parts
        )

        record_id = str(
            meta.get(
                "id",
                "",
            )
        ).strip()

        if not record_id:
            record_id = (
                f"core:{line_number}"
            )

        result.append(
            {
                "id": record_id,
                "question": question,
                "answer": answer,
                "embedding_text": (
                    embedding_text
                ),
                "meta": {
                    "id": record_id,
                    "question": question,
                    "aliases": aliases,
                    "tags": meta.get(
                        "tags",
                        [],
                    ),
                    "source": (
                        "core_knowledge"
                    ),
                    "source_type": "core",
                    "line": meta.get(
                        "line",
                        line_number,
                    ),
                },
            }
        )

    core_count = len(
        result
    )

    # --------------------------------------------------------
    # TECHNICAL
    #
    # В embedding идет ТОЛЬКО вопрос.
    # Ответ хранится как retrieval text.
    # --------------------------------------------------------

    technical_records = read_index_jsonl(
        TECHNICAL_JSONL
    )

    technical_added = 0

    for answer, meta, line_number in (
        technical_records
    ):
        question = str(
            meta.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            raise RuntimeError(
                "TECHNICAL record без question: "
                f"строка {line_number}"
            )

        normalized = normalize_question(
            question
        )

        if normalized in seen_questions:
            skipped_duplicates += 1
            continue

        seen_questions.add(
            normalized
        )

        source_line = meta.get(
            "line",
            line_number,
        )

        record_id = (
            f"technical:{source_line}"
        )

        result.append(
            {
                "id": record_id,
                "question": question,
                "answer": answer,
                "embedding_text": (
                    f"Вопрос: {question}"
                ),
                "meta": {
                    "id": record_id,
                    "question": question,
                    "source": (
                        "nova_technical_v1_1"
                    ),
                    "source_type": (
                        "technical"
                    ),
                    "line": source_line,
                },
            }
        )

        technical_added += 1

    return (
        result,
        core_count,
        technical_added,
        skipped_duplicates,
    )


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
            "Build consolidated trusted "
            "RAG index for NOVA."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить перестроить "
            "существующий trusted_index."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "NOVA — BUILD TRUSTED RAG INDEX"
    )
    print("=" * 80)
    print()

    print(
        f"CORE:      {CORE_JSONL}"
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
        "Embedding:"
    )

    print(
        "  CORE      = question + aliases"
    )

    print(
        "  TECHNICAL = question only"
    )

    print(
        "  answers are NOT embedded"
    )

    print(
        "Pooling: masked_mean"
    )

    print()

    if not CORE_JSONL.exists():
        raise FileNotFoundError(
            CORE_JSONL
        )

    if not TECHNICAL_JSONL.exists():
        raise FileNotFoundError(
            TECHNICAL_JSONL
        )

    if output_exists():
        if not args.force:
            raise RuntimeError(
                "trusted_index уже существует.\n"
                "Ничего не перезаписано.\n"
                "Для rebuild используй --force."
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

    core_hash = sha256_file(
        CORE_JSONL
    )

    technical_hash = sha256_file(
        TECHNICAL_JSONL
    )

    print(
        f"CORE SHA-256:      {core_hash}"
    )

    print(
        f"TECHNICAL SHA-256: "
        f"{technical_hash}"
    )

    print()
    print(
        "Reading trusted records..."
    )

    (
        records,
        core_count,
        technical_count,
        skipped_duplicates,
    ) = load_trusted_records()

    total = len(
        records
    )

    print(
        f"CORE added:       "
        f"{core_count:,}"
    )

    print(
        f"TECHNICAL added:  "
        f"{technical_count:,}"
    )

    print(
        f"Duplicates skip:  "
        f"{skipped_duplicates:,}"
    )

    print(
        f"TOTAL trusted:    "
        f"{total:,}"
    )

    if total == 0:
        raise RuntimeError(
            "Trusted dataset пуст."
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
        "Building embeddings..."
    )

    started_embeddings = time.time()

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

        embedding_texts = [
            item[
                "embedding_text"
            ]
            for item in batch
        ]

        answers = [
            item[
                "answer"
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
                embedding_texts,
                batch_size=BATCH_SIZE,
            )
        )

        store.add(
            texts=answers,
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
            f"| {rate:,.1f} records/s",
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

    if index.d != embedder.dim:
        raise RuntimeError(
            "Неверная размерность "
            "trusted index."
        )

    if index.ntotal != total:
        raise RuntimeError(
            "FAISS ntotal не совпадает "
            "с количеством trusted records."
        )

    if jsonl_count != total:
        raise RuntimeError(
            "JSONL count не совпадает "
            "с количеством trusted records."
        )

    print(
        "Temporary index: OK"
    )

    final_faiss = Path(
        str(OUTPUT_BASE)
        + ".faiss"
    )

    final_jsonl = Path(
        str(OUTPUT_BASE)
        + ".jsonl"
    )

    print()
    print(
        "Moving temporary index "
        "to final paths..."
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
        "NOVA TRUSTED RAG BUILD REPORT",
        "=" * 80,
        "",
        f"CORE source: {CORE_JSONL}",
        f"CORE SHA-256: {core_hash}",
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
        "",
        (
            "CORE embedding: "
            "question + aliases"
        ),
        (
            "TECHNICAL embedding: "
            "question only"
        ),
        "Answers embedded: NO",
        "",
        f"CORE added: {core_count}",
        (
            "TECHNICAL added: "
            f"{technical_count}"
        ),
        (
            "Duplicates skipped: "
            f"{skipped_duplicates}"
        ),
        f"TOTAL: {total}",
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
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "TRUSTED RAG BUILD COMPLETE"
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

    print()
    print(
        "Existing CORE / TECHNICAL / "
        "BROAD indexes were NOT modified."
    )


if __name__ == "__main__":
    main()