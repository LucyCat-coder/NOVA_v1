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


SOURCE_JSONL = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "core_knowledge.jsonl"
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "rag"
    / "core_index"
)

TEMP_BASE = (
    PROJECT_ROOT
    / "rag"
    / "core_index_building"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "core_rag_build_report.txt"
)


MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

BATCH_SIZE = 32


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


def load_core(
    path: Path,
):
    answers = []
    metadata = []
    embedding_texts = []

    seen_ids = set()

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

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Некорректный JSON "
                    f"в строке {line_number}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "ожидался JSON object."
                )

            record_id = str(
                record.get(
                    "id",
                    "",
                )
            ).strip()

            question = str(
                record.get(
                    "question",
                    "",
                )
            ).strip()

            answer = str(
                record.get(
                    "answer",
                    "",
                )
            ).strip()

            aliases = record.get(
                "aliases",
                [],
            )

            tags = record.get(
                "tags",
                [],
            )

            if not record_id:
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "нет id."
                )

            if record_id in seen_ids:
                raise RuntimeError(
                    "Дублирующийся id: "
                    f"{record_id}"
                )

            seen_ids.add(
                record_id
            )

            if not question:
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "нет question."
                )

            if not answer:
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "нет answer."
                )

            if not isinstance(
                aliases,
                list,
            ):
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "aliases должен быть list."
                )

            if not isinstance(
                tags,
                list,
            ):
                raise RuntimeError(
                    f"Строка {line_number}: "
                    "tags должен быть list."
                )

            aliases = [
                str(alias).strip()
                for alias in aliases
                if str(alias).strip()
            ]

            tags = [
                str(tag).strip()
                for tag in tags
                if str(tag).strip()
            ]

            # ВАЖНО:
            # embedding строим по вопросу
            # и его алиасам, но НЕ по ответу.
            #
            # Это уменьшает размывание intent.
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

            answers.append(
                answer
            )

            metadata.append(
                {
                    "id": record_id,
                    "question": question,
                    "aliases": aliases,
                    "tags": tags,
                    "source": "core_knowledge",
                    "source_type": "core",
                    "line": line_number,
                }
            )

            embedding_texts.append(
                embedding_text
            )

    return (
        answers,
        metadata,
        embedding_texts,
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
            "Build NOVA trusted CORE RAG index."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить перестроить "
            "существующий core_index."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "NOVA — BUILD CORE RAG INDEX"
    )
    print("=" * 80)
    print()

    print(
        f"Source: {SOURCE_JSONL}"
    )

    print(
        f"Output: "
        f"{OUTPUT_BASE}.{{faiss,jsonl}}"
    )

    print(
        f"Model:  {MODEL_NAME}"
    )

    print(
        "Pooling: masked_mean"
    )

    print(
        "Embedding:"
    )

    print(
        "  canonical question + aliases"
    )

    print(
        "  answer is NOT embedded"
    )

    print()

    if not SOURCE_JSONL.exists():
        raise FileNotFoundError(
            SOURCE_JSONL
        )

    if output_exists():
        if not args.force:
            raise RuntimeError(
                "core_index уже существует.\n"
                "Ничего не перезаписано.\n"
                "Для rebuild используй --force."
            )

        print(
            "Removing existing core_index..."
        )

        remove_files(
            OUTPUT_BASE
        )

    remove_files(
        TEMP_BASE
    )

    started_total = time.time()

    print(
        "Calculating source SHA-256..."
    )

    source_hash = sha256_file(
        SOURCE_JSONL
    )

    print(
        f"Source SHA-256: "
        f"{source_hash}"
    )

    print()
    print(
        "Reading CORE knowledge..."
    )

    (
        answers,
        metadata,
        embedding_texts,
    ) = load_core(
        SOURCE_JSONL
    )

    total = len(
        answers
    )

    print(
        f"CORE records: {total:,}"
    )

    if total == 0:
        raise RuntimeError(
            "core_knowledge.jsonl пуст."
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

        embeddings = (
            embedder.encode_passages(
                embedding_texts[
                    start:end
                ],
                batch_size=BATCH_SIZE,
            )
        )

        # Возвращаемый retrieval text —
        # именно проверенный CORE answer.
        store.add(
            texts=answers[
                start:end
            ],
            embeddings=embeddings,
            metadata=metadata[
                start:end
            ],
        )

        print(
            f"\rProcessed "
            f"{end:,}/{total:,}",
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
            "core index."
        )

    if index.ntotal != total:
        raise RuntimeError(
            "FAISS ntotal не совпадает "
            "с числом CORE records."
        )

    if jsonl_count != total:
        raise RuntimeError(
            "JSONL count не совпадает "
            "с числом CORE records."
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
        "NOVA CORE RAG BUILD REPORT",
        "=" * 80,
        "",
        f"Source: {SOURCE_JSONL}",
        f"Source SHA-256: {source_hash}",
        "",
        f"Model: {MODEL_NAME}",
        "Pooling: masked_mean",
        (
            "Embedding: canonical question "
            "+ aliases"
        ),
        "Answer embedded: NO",
        "",
        f"Records: {total}",
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
        "CORE RAG BUILD COMPLETE"
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
        "No existing RAG index "
        "was modified."
    )


if __name__ == "__main__":
    main()