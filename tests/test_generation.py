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


def main():
    print()
    print("=" * 78)
    print("NOVA FINAL — GENERATION TEST — RAG OFF")
    print("=" * 78)

    assistant = Assistant(
        checkpoint_path=RUNTIME_CONFIG["checkpoint"],
        model_config=MODEL_CONFIG,
        device=RUNTIME_CONFIG["device"],
        rag_index_path=None,
    )

    print()
    print("Параметры генерации:")
    print(
        f"  temperature      = "
        f"{RUNTIME_CONFIG['temperature']}"
    )
    print(
        f"  top_k            = "
        f"{RUNTIME_CONFIG['top_k']}"
    )
    print(
        f"  top_p            = "
        f"{RUNTIME_CONFIG['top_p']}"
    )
    print(
        f"  repetition       = "
        f"{RUNTIME_CONFIG['repetition_penalty']}"
    )
    print("  RAG              = OFF")
    print("  seed             = 1337")

    print()

    started_all = time.time()

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print("=" * 78)
        print(
            f"[{index}/{len(QUESTIONS)}]"
        )
        print(
            f"Пользователь: {question}"
        )
        print("-" * 78)

        started = time.time()

        answer = assistant.generate(
            question,
            max_new_tokens=120,
            temperature=RUNTIME_CONFIG[
                "temperature"
            ],
            top_k=RUNTIME_CONFIG[
                "top_k"
            ],
            top_p=RUNTIME_CONFIG[
                "top_p"
            ],
            repetition_penalty=RUNTIME_CONFIG[
                "repetition_penalty"
            ],
            seed=1337,
            rag_enabled=False,
        )

        elapsed = (
            time.time()
            - started
        )

        print(
            f"Нова: {answer}"
        )

        print()
        print(
            f"Время: {elapsed:.2f}s"
        )

    print()
    print("=" * 78)
    print(
        "Готово за "
        f"{time.time() - started_all:.1f}s"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()