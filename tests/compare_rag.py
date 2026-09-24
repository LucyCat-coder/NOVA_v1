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


from config import (  # noqa: E402
    MODEL_CONFIG,
    RUNTIME_CONFIG,
)

from inference import Assistant  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_rag_comparison.txt"
)


QUESTIONS = [
    "Сколько будет 2 + 2?",

    "Объясни простыми словами разницу между TCP и UDP.",

    "Чем Docker-контейнер отличается от виртуальной машины?",

    "Почему нельзя использовать изменяемый объект как значение аргумента по умолчанию в Python?",

    "В чём разница между git merge и git rebase?",

    "Как заполнить пропущенные значения в pandas DataFrame?",

    "Что такое PostgreSQL?",

    "Как работает индекс в базе данных?",

    "Что такое REST API?",
]


def short_context(
    text: str,
    limit: int = 900,
) -> str:
    text = (
        text
        .replace("\r", " ")
        .strip()
    )

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def main():
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("NOVA FINAL — RAG OFF vs INLINE")
    print("=" * 80)

    assistant = Assistant(
        checkpoint_path=RUNTIME_CONFIG[
            "checkpoint"
        ],
        model_config=MODEL_CONFIG,
        device=RUNTIME_CONFIG[
            "device"
        ],
        rag_index_path=RUNTIME_CONFIG[
            "rag_index"
        ],
    )

    report = []

    def write(
        text: str = "",
    ):
        print(text)
        report.append(text)

    write()
    write("=" * 100)
    write("NOVA FINAL — RAG OFF vs INLINE")
    write("=" * 100)
    write()
    write("Generation:")
    write("temperature        = 0.0")
    write("top_k              = OFF")
    write("top_p              = OFF")
    write("repetition penalty = 1.0")
    write("max_new_tokens      = 160")
    write()
    write("RAG:")
    write("mode                = inline")
    write("top_k_docs          = 1")
    write("min_score           = NONE")
    write("reranker             = OFF")
    write()

    started_all = time.time()

    for number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        write("=" * 100)
        write(
            f"QUESTION {number}/{len(QUESTIONS)}"
        )
        write("=" * 100)
        write(
            f"Пользователь: {question}"
        )
        write()

        started = time.time()

        answer_off = assistant.generate(
            question,
            max_new_tokens=160,
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=None,
            rag_enabled=False,
        )

        elapsed_off = (
            time.time()
            - started
        )

        write("--- RAG OFF ---")
        write(answer_off)
        write(
            f"[time: {elapsed_off:.3f}s]"
        )
        write()

        started = time.time()

        answer_rag, contexts = (
            assistant.generate(
                question,
                max_new_tokens=160,
                temperature=0.0,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0,
                seed=None,
                rag_enabled=True,
                rag_mode="inline",
                top_k_docs=1,

                # ВАЖНО:
                # пока без threshold.
                min_score=None,

                use_reranker=False,
                return_context=True,
            )
        )

        elapsed_rag = (
            time.time()
            - started
        )

        write("--- RETRIEVED CONTEXT ---")

        if contexts:
            write(
                short_context(
                    contexts[0]
                )
            )
        else:
            write(
                "(no context)"
            )

        write()

        write("--- RAG INLINE ---")
        write(answer_rag)
        write(
            f"[time: {elapsed_rag:.3f}s]"
        )
        write()

    write("=" * 100)
    write(
        "Total time: "
        f"{time.time() - started_all:.1f}s"
    )
    write("=" * 100)

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print(
        f"Отчёт сохранён: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()