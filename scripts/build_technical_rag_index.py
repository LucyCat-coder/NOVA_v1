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
    / "nova_technical_v1_1"
    / "nova_technical_v1_1_prepared.jsonl"
)

OUTPUT_BASE = (
    PROJECT_ROOT
    / "rag"
    / "technical_index"
)

TEMP_BASE = (
    PROJECT_ROOT
    / "rag"
    / "technical_index_building"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "technical_rag_build_report.txt"
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


def extract_qa(
    record: dict,
    line_number: int,
) -> tuple[str, str]:
    messages = record.get(
        "messages"
    )

    if not isinstance(
        messages,
        list,
    ):
        raise RuntimeError(
            f"Строка {line_number}: "
            "нет списка messages."
        )

    user_parts = []
    assistant_parts = []

    for message in messages:
        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip().lower()

        content = message.get(
            "content",
            "",
        )

        if isinstance(
            content,
            list,
        ):
            pieces = []

            for item in content:
                if isinstance(
                    item,
                    str,
                ):
                    pieces.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):
                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        pieces.append(
                            str(text)
                        )

            content = "\n".join(
                pieces
            )

        content = str(
            content
        ).strip()

        if not content:
            continue

        if role in {
            "user",
            "human",
        }:
            user_parts.append(
                content
            )

        elif role in {
            "assistant",
            "bot",
            "model",
            "gpt",
        }:
            assistant_parts.append(
                content
            )

    question = "\n".join(
        user_parts
    ).strip()

    answer = "\n".join(
        assistant_parts
    ).strip()

    if not question:
        raise RuntimeError(
            f"Строка {line_number}: "
            "не найден user question."
        )

    if not answer:
        raise RuntimeError(
            f"Строка {line_number}: "
            "не найден assistant answer."
        )

    return (
        question,
        answer,
    )


def load_dataset(
    path: Path,
):
    questions = []
    answers = []
    metadata = []
    embedding_texts = []

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
                    "JSON record должен быть object."
                )

            question, answer = (
                extract_qa(
                    record,
                    line_number,
                )
            )

            questions.append(
                question
            )

            answers.append(
                answer
            )

            metadata.append(
                {
                    "question": question,
                    "source": (
                        "nova_technical_v1_1"
                    ),
                    "line": line_number,
                    "chunk_id": 0,
                }
            )

            embedding_texts.append(
                (
                    f"Вопрос: {question}\n"
                    f"Ответ: {answer}"
                )
            )

    return (
        questions,
        answers,
        metadata,
        embedding_texts,
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


def remove_final_files():
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
            "Build trusted technical "
            "RAG index for NOVA."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Разрешить перестроить "
            "существующий technical index."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "NOVA — BUILD TECHNICAL RAG INDEX"
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
        "  Вопрос: <question>"
    )

    print(
        "  Ответ: <answer>"
    )

    print()

    if not SOURCE_JSONL.exists():
        raise FileNotFoundError(
            SOURCE_JSONL
        )

    if final_files_exist():
        if not args.force:
            raise RuntimeError(
                "technical_index уже существует.\n"
                "Ничего не перезаписано.\n"
                "Для осознанного rebuild "
                "используй --force."
            )

        remove_final_files()

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
        "Reading technical dataset..."
    )

    (
        questions,
        answers,
        metadata,
        embedding_texts,
    ) = load_dataset(
        SOURCE_JSONL
    )

    total_records = len(
        answers
    )

    print(
        f"Records: {total_records:,}"
    )

    if total_records == 0:
        raise RuntimeError(
            "Technical dataset пуст."
        )

    if not (
        len(questions)
        == len(answers)
        == len(metadata)
        == len(embedding_texts)
    ):
        raise RuntimeError(
            "Внутренняя ошибка: "
            "размеры dataset arrays "
            "не совпадают."
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

    started_embeddings = (
        time.time()
    )

    for start in range(
        0,
        total_records,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            total_records,
        )

        embeddings = (
            embedder.encode_passages(
                embedding_texts[
                    start:end
                ],
                batch_size=BATCH_SIZE,
            )
        )

        # В FAISS embedding строится по
        # question + answer.
        #
        # Но при retrieval пользователю
        # возвращаем только clean answer.
        store.add(
            texts=answers[
                start:end
            ],
            embeddings=embeddings,
            metadata=metadata[
                start:end
            ],
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
            f"{end:,}/{total_records:,} "
            f"({end / total_records * 100:5.1f}%) "
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

    if (
        index.d
        != embedder.dim
    ):
        raise RuntimeError(
            "Неверная размерность "
            "technical index."
        )

    if (
        index.ntotal
        != total_records
    ):
        raise RuntimeError(
            "FAISS содержит неверное "
            "число technical records."
        )

    if (
        jsonl_count
        != total_records
    ):
        raise RuntimeError(
            "JSONL содержит неверное "
            "число technical records."
        )

    print(
        "Temporary index: OK"
    )

    print()
    print(
        "Moving temporary index "
        "to final paths..."
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

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = [
        "NOVA TECHNICAL RAG BUILD REPORT",
        "=" * 80,
        "",
        f"Source: {SOURCE_JSONL}",
        f"Source SHA-256: {source_hash}",
        "",
        f"Model: {MODEL_NAME}",
        "Pooling: masked_mean",
        "",
        "Embedding format:",
        "  Вопрос: <question>",
        "  Ответ: <answer>",
        "",
        f"Records: {total_records}",
        f"Embedding dimension: {index.d}",
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
        "TECHNICAL RAG BUILD COMPLETE"
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
        "knowledge_index_v2.* "
        "was NOT modified."
    )


if __name__ == "__main__":
    main()