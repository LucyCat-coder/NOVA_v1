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


from config import (  # noqa: E402
    CHECKPOINTS,
    MODEL_CONFIG,
)

from inference import Assistant  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison_no_rag.txt"
)


QUESTIONS = [
    "Сколько будет 2 + 2?",

    "Какого цвета ясное дневное небо?",

    "У Маши было 12 яблок. Она отдала 5. Сколько яблок осталось?",

    "Объясни простыми словами разницу между TCP и UDP.",

    "Чем Docker-контейнер отличается от виртуальной машины?",

    "Почему изменяемый объект нельзя безопасно использовать как значение аргумента по умолчанию в Python?",

    "В чём разница между git merge и git rebase?",

    "Как заполнить пропущенные значения в pandas DataFrame?",
]


MODELS = [
    (
        "NOVA FINAL",
        CHECKPOINTS["final"],
    ),
    (
        "NOVA V1",
        CHECKPOINTS["nova_v1"],
    ),
    (
        "TECHNICAL V1.1",
        CHECKPOINTS["technical_v1_1"],
    ),
]


def append(
    lines: list[str],
    text: str = "",
):
    lines.append(text)
    print(text)


def unload_assistant(
    assistant,
):
    if assistant is not None:
        try:
            assistant.model.to("cpu")
        except Exception:
            pass

        del assistant

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(
    model_name: str,
    checkpoint_path: Path,
):
    assistant = None
    results = []

    try:
        assistant = Assistant(
            checkpoint_path=checkpoint_path,
            model_config=MODEL_CONFIG,
            device="cuda",
            rag_index_path=None,
        )

        for index, question in enumerate(
            QUESTIONS,
            start=1,
        ):
            started = time.time()

            answer = assistant.generate(
                question,
                max_new_tokens=120,

                # ВАЖНО:
                # greedy generation.
                #
                # Так мы убираем случайность sampling
                # из сравнения трёх моделей.
                temperature=0.0,

                top_k=None,
                top_p=None,

                # Для greedy comparison сначала
                # вообще отключаем penalty.
                repetition_penalty=1.0,

                seed=None,
                rag_enabled=False,
            )

            elapsed = (
                time.time()
                - started
            )

            results.append(
                {
                    "index": index,
                    "question": question,
                    "answer": answer,
                    "elapsed": elapsed,
                }
            )

    finally:
        unload_assistant(
            assistant
        )

    return results


def main():
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    total_started = time.time()

    print()
    print("=" * 80)
    print("NOVA — MODEL COMPARISON — RAG OFF — GREEDY")
    print("=" * 80)
    print()
    print("temperature        = 0.0")
    print("top_k              = OFF")
    print("top_p              = OFF")
    print("repetition penalty = 1.0")
    print("RAG                 = OFF")
    print()

    for model_name, checkpoint_path in MODELS:
        print()
        print("#" * 80)
        print(model_name)
        print("#" * 80)

        results = run_model(
            model_name,
            checkpoint_path,
        )

        all_results[
            model_name
        ] = results

        for item in results:
            print()
            print(
                f"[{item['index']}/{len(QUESTIONS)}] "
                f"{item['question']}"
            )
            print("-" * 80)
            print(item["answer"])
            print(
                f"[{item['elapsed']:.2f}s]"
            )

    report: list[str] = []

    append(
        report,
        "=" * 100,
    )

    append(
        report,
        "NOVA MODEL COMPARISON — RAG OFF — GREEDY",
    )

    append(
        report,
        "=" * 100,
    )

    append(report)

    append(
        report,
        "Generation settings:",
    )

    append(
        report,
        "temperature        = 0.0",
    )

    append(
        report,
        "top_k              = OFF",
    )

    append(
        report,
        "top_p              = OFF",
    )

    append(
        report,
        "repetition penalty = 1.0",
    )

    append(
        report,
        "RAG                 = OFF",
    )

    append(report)

    for question_index, question in enumerate(
        QUESTIONS
    ):
        append(
            report,
            "=" * 100,
        )

        append(
            report,
            f"QUESTION {question_index + 1}",
        )

        append(
            report,
            "=" * 100,
        )

        append(
            report,
            f"Пользователь: {question}",
        )

        append(report)

        for model_name, _ in MODELS:
            item = all_results[
                model_name
            ][
                question_index
            ]

            append(
                report,
                f"--- {model_name} ---",
            )

            append(
                report,
                item["answer"],
            )

            append(
                report,
                f"[time: {item['elapsed']:.2f}s]",
            )

            append(report)

    append(
        report,
        "=" * 100,
    )

    append(
        report,
        "CHECKPOINTS",
    )

    append(
        report,
        "=" * 100,
    )

    for model_name, checkpoint_path in MODELS:
        append(
            report,
            f"{model_name}: {checkpoint_path}",
        )

    append(report)

    append(
        report,
        "Total time: "
        f"{time.time() - total_started:.1f}s",
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        f"Отчёт сохранён: {REPORT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()