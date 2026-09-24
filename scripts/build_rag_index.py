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


from rag import (  # noqa: E402
    Embedder,
    VectorStore,
)


SOURCE_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index.jsonl"
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2"
)

TEMP_BASE = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2_building"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rag_v2_build_report.txt"
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

    with path.open(
        "rb"
    ) as file:
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


def load_source(
    path: Path,
):
    texts = []
    metadata = []
    embedding_texts = []

    missing_question = 0

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

            if "text" not in record:
                raise RuntimeError(
                    "Нет поля 'text' "
                    f"в строке {line_number}"
                )

            text = str(
                record["text"]
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
                    "meta должен быть dict, "
                    f"строка {line_number}"
                )

            question = str(
                meta.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                missing_question += 1

            texts.append(
                text
            )

            metadata.append(
                dict(meta)
            )

            if question:
                embedding_text = (
                    f"Вопрос: {question}\n"
                    f"Ответ: {text}"
                )
            else:
                embedding_text = (
                    f"Ответ: {text}"
                )

            embedding_texts.append(
                embedding_text
            )

    return (
        texts,
        metadata,
        embedding_texts,
        missing_question,
    )


def remove_temp_files():
    for suffix in [
        ".faiss",
        ".jsonl",
    ]:
        path = Path(
            str(TEMP_BASE)
            + suffix
        )

        if path.exists():
            path.unlink()


def final_files_exist() -> bool:
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


def move_temp_to_final():
    temp_faiss = Path(
        str(TEMP_BASE)
        + ".faiss"
    )

    temp_jsonl = Path(
        str(TEMP_BASE)
        + ".jsonl"
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build NOVA RAG v2 index."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить удалить уже "
            "существующий v2 index."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("NOVA — BUILD RAG INDEX V2")
    print("=" * 80)
    print()

    print(
        f"Source: {SOURCE_JSONL}"
    )

    print(
        f"Output: {OUTPUT_BASE}.{{faiss,jsonl}}"
    )

    print(
        f"Model:  {MODEL_NAME}"
    )

    print(
        "Pooling: masked_mean"
    )

    print(
        "Embedding text:"
    )

    print(
        "  Вопрос: <original question>"
    )

    print(
        "  Ответ: <original chunk>"
    )

    print()

    if not SOURCE_JSONL.exists():
        raise FileNotFoundError(
            SOURCE_JSONL
        )

    if final_files_exist():
        if not args.force:
            raise RuntimeError(
                "knowledge_index_v2 уже существует.\n"
                "Скрипт ничего не перезаписал.\n"
                "Если действительно нужен rebuild, "
                "запусти с --force."
            )

        for suffix in [
            ".faiss",
            ".jsonl",
        ]:
            path = Path(
                str(OUTPUT_BASE)
                + suffix
            )

            if path.exists():
                print(
                    f"Removing old: {path}"
                )

                path.unlink()

    remove_temp_files()

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
        "Reading source metadata..."
    )

    (
        texts,
        metadata,
        embedding_texts,
        missing_question,
    ) = load_source(
        SOURCE_JSONL
    )

    total_records = len(
        texts
    )

    print(
        f"Records: {total_records:,}"
    )

    print(
        "Missing question metadata: "
        f"{missing_question:,}"
    )

    if total_records == 0:
        raise RuntimeError(
            "Source index пуст."
        )

    if missing_question > 0:
        print()
        print(
            "[WARN] Некоторые chunks "
            "не имеют исходного question."
        )

        print(
            "[WARN] Для них embedding "
            "будет построен только по answer."
        )

    print()
    print(
        "Loading multilingual-e5-small..."
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

    started_embeddings = (
        time.time()
    )

    processed = 0

    for start in range(
        0,
        total_records,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            total_records,
        )

        batch_embedding_texts = (
            embedding_texts[
                start:end
            ]
        )

        batch_embeddings = (
            embedder.encode_passages(
                batch_embedding_texts,
                batch_size=BATCH_SIZE,
            )
        )

        # В индекс попадают embeddings
        # question + answer.
        #
        # Но пользователю при retrieval
        # возвращаем только оригинальный answer.
        store.add(
            texts=texts[
                start:end
            ],
            embeddings=batch_embeddings,
            metadata=metadata[
                start:end
            ],
        )

        processed = end

        if (
            processed % 1000 < BATCH_SIZE
            or processed == total_records
        ):
            elapsed = (
                time.time()
                - started_embeddings
            )

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"\rProcessed "
                f"{processed:,}"
                f"/{total_records:,} "
                f"({processed / total_records * 100:5.1f}%) "
                f"| {rate:,.1f} chunks/s",
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

    jsonl_count = 0

    with temp_jsonl.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                jsonl_count += 1

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
            "готового индекса."
        )

    if (
        index.ntotal
        != total_records
    ):
        raise RuntimeError(
            "FAISS содержит неправильное "
            "число записей."
        )

    if (
        jsonl_count
        != total_records
    ):
        raise RuntimeError(
            "JSONL содержит неправильное "
            "число записей."
        )

    print(
        "Temporary index: OK"
    )

    print()
    print(
        "Moving temporary index "
        "to final v2 paths..."
    )

    move_temp_to_final()

    final_faiss = Path(
        str(OUTPUT_BASE)
        + ".faiss"
    )

    final_jsonl = Path(
        str(OUTPUT_BASE)
        + ".jsonl"
    )

    total_time = (
        time.time()
        - started_total
    )

    report = [
        "NOVA RAG V2 BUILD REPORT",
        "=" * 80,
        "",
        f"Source: {SOURCE_JSONL}",
        f"Source SHA-256: {source_hash}",
        "",
        f"Model: {MODEL_NAME}",
        "Pooling: masked_mean",
        "Embedding format:",
        "  Вопрос: <question>",
        "  Ответ: <chunk>",
        "",
        f"Records: {total_records}",
        (
            "Missing question metadata: "
            f"{missing_question}"
        ),
        f"Embedding dimension: {index.d}",
        "",
        (
            "Output FAISS: "
            f"{final_faiss}"
        ),
        (
            "Output JSONL: "
            f"{final_jsonl}"
        ),
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

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RAG V2 BUILD COMPLETE")
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
        "Original knowledge_index.* "
        "was NOT modified."
    )


if __name__ == "__main__":
    main()