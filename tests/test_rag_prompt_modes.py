from __future__ import annotations

import re
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


from config import MODEL_CONFIG, RUNTIME_CONFIG  # noqa: E402
from inference import Assistant  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rag_prompt_modes_test.txt"
)


MAX_NEW_TOKENS = 160


TESTS = [
    {
        "name": "POSTGRESQL",
        "query": (
            "Postgres — это язык, "
            "база данных или СУБД?"
        ),
        "rag_source": "auto",
    },

    {
        "name": "PYTHON MUTABLE DEFAULT",
        "query": (
            "Почему список в "
            "default-аргументе Python "
            "сохраняет изменения "
            "между вызовами?"
        ),
        "rag_source": "auto",
    },

    {
        "name": "REST",
        "query": (
            "REST — это просто HTTP "
            "с JSON или что-то другое?"
        ),
        "rag_source": "auto",
    },

    {
        "name": "DOCKER STOP VS KILL",
        "query": (
            "Чем docker stop "
            "отличается от docker kill?"
        ),
        "rag_source": "technical",
    },
]


GENERATED_MODES = [
    "plain",
    "inline",
    "naive",
    "prefix",
    "fewshot",
]


def normalize_words(
    text: str,
) -> set[str]:
    return set(
        re.findall(
            r"[a-zа-яё0-9_+-]+",
            text.lower(),
            flags=re.IGNORECASE,
        )
    )


def context_overlap(
    answer: str,
    context: str,
) -> float:
    answer_words = normalize_words(
        answer
    )

    context_words = normalize_words(
        context
    )

    if not answer_words:
        return 0.0

    return (
        len(
            answer_words
            & context_words
        )
        / len(
            answer_words
        )
    )


def decode_clean(
    assistant: Assistant,
    token_ids: list[int],
) -> str:
    """
    decode_bytes + errors='ignore'
    убирает артефакт �, если generation
    оборвалась на части UTF-8 sequence.
    """

    raw = (
        assistant.tokenizer
        .decode_bytes(
            token_ids
        )
    )

    text = raw.decode(
        "utf-8",
        errors="ignore",
    )

    return assistant._clean_answer(
        text
    )


def generate_from_prompt(
    assistant: Assistant,
    prompt: str,
) -> tuple[str, int]:
    max_prompt_tokens = (
        assistant.block_size
        - MAX_NEW_TOKENS
    )

    input_ids = (
        assistant.tokenizer.encode(
            prompt,
            disallowed_special=(),
        )
    )

    if (
        len(input_ids)
        > max_prompt_tokens
    ):
        input_ids = input_ids[
            -max_prompt_tokens:
        ]

    new_ids = (
        assistant._generate_tokens(
            input_ids=input_ids,
            max_new_tokens=(
                MAX_NEW_TOKENS
            ),
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=None,
        )
    )

    answer = decode_clean(
        assistant,
        new_ids,
    )

    return (
        answer,
        len(input_ids),
    )


def build_prompt(
    assistant: Assistant,
    *,
    mode: str,
    query: str,
    context: str,
    source_question: str,
) -> str:
    max_prompt_tokens = (
        assistant.block_size
        - MAX_NEW_TOKENS
    )

    if mode == "plain":
        return assistant._plain_prompt(
            query
        )

    if mode in {
        "inline",
        "naive",
        "prefix",
    }:
        return (
            assistant._build_rag_prompt(
                query=query,
                contexts=[
                    context
                ],
                mode=mode,
                max_prompt_tokens=(
                    max_prompt_tokens
                ),
            )
        )

    if mode == "fewshot":
        # Максимально близко к формату,
        # который модель уже видела:
        #
        # Пользователь: source question
        # Нова: source answer
        #
        # Пользователь: actual query
        # Нова:
        return (
            f"Пользователь: "
            f"{source_question}\n"
            f"Нова: {context}\n\n"
            f"Пользователь: {query}\n"
            f"Нова:"
        )

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def compact(
    text: str,
    limit: int = 2500,
) -> str:
    text = str(
        text
    ).strip()

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — RAG PROMPT MODES TEST"
    )
    print("=" * 100)
    print()

    assistant = Assistant(
        checkpoint_path=(
            RUNTIME_CONFIG[
                "checkpoint"
            ]
        ),
        model_config=MODEL_CONFIG,
        device=(
            RUNTIME_CONFIG[
                "device"
            ]
        ),
        rag_core_index_path=(
            RUNTIME_CONFIG[
                "rag_core_index"
            ]
        ),
        rag_technical_index_path=(
            RUNTIME_CONFIG[
                "rag_technical_index"
            ]
        ),
        rag_broad_index_path=(
            RUNTIME_CONFIG[
                "rag_broad_index"
            ]
        ),
        rag_core_threshold=(
            RUNTIME_CONFIG[
                "rag_core_threshold"
            ]
        ),
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA RAG PROMPT MODES TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Generation:"
    )

    lines.append(
        "  temperature = 0"
    )

    lines.append(
        "  top_k = OFF"
    )

    lines.append(
        "  top_p = OFF"
    )

    lines.append(
        "  repetition_penalty = 1"
    )

    lines.append(
        f"  max_new_tokens = "
        f"{MAX_NEW_TOKENS}"
    )

    lines.append("")

    lines.append(
        "Modes:"
    )

    for mode in GENERATED_MODES:
        lines.append(
            f"  {mode}"
        )

    lines.append(
        "  direct_retrieval "
        "(no generation)"
    )

    lines.append("")

    started_all = time.time()

    for test_number, test in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{test_number}/"
            f"{len(TESTS)}] "
            f"{test['name']}"
        )

        query = test[
            "query"
        ]

        rag_source = test[
            "rag_source"
        ]

        retrieval = (
            assistant.safe_retriever
            .retrieve_with_info(
                query=query,
                mode=rag_source,
                top_k=1,
                min_score=0.80,
            )
        )

        results = retrieval[
            "results"
        ]

        if not results:
            raise RuntimeError(
                "Для теста не найден "
                "retrieval context: "
                f"{test['name']}"
            )

        context = results[
            0
        ][
            0
        ]

        score = float(
            results[
                0
            ][
                1
            ]
        )

        metadata = results[
            0
        ][
            2
        ]

        source_question = str(
            metadata.get(
                "question",
                query,
            )
        ).strip()

        lines.append(
            "=" * 100
        )

        lines.append(
            f"TEST {test_number}: "
            f"{test['name']}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"RAG SOURCE: "
            f"{rag_source}"
        )

        lines.append(
            f"ROUTE: "
            f"{retrieval.get('route')}"
        )

        lines.append(
            f"SCORE: "
            f"{score:.6f}"
        )

        lines.append(
            f"SOURCE QUESTION: "
            f"{source_question}"
        )

        lines.append("")

        lines.append(
            "DIRECT RETRIEVAL BASELINE:"
        )

        lines.append(
            compact(
                context
            )
        )

        lines.append("")

        for mode in GENERATED_MODES:
            prompt = build_prompt(
                assistant,
                mode=mode,
                query=query,
                context=context,
                source_question=(
                    source_question
                ),
            )

            started = time.time()

            (
                answer,
                prompt_tokens,
            ) = generate_from_prompt(
                assistant,
                prompt,
            )

            elapsed = (
                time.time()
                - started
            )

            overlap = (
                context_overlap(
                    answer,
                    context,
                )
            )

            lines.append(
                "-" * 100
            )

            lines.append(
                f"MODE: {mode}"
            )

            lines.append(
                f"PROMPT TOKENS: "
                f"{prompt_tokens}"
            )

            lines.append(
                f"TIME: "
                f"{elapsed:.3f}s"
            )

            lines.append(
                "CONTEXT WORD OVERLAP: "
                f"{overlap:.3f}"
            )

            lines.append(
                "ANSWER:"
            )

            if answer:
                lines.append(
                    compact(
                        answer
                    )
                )

            else:
                lines.append(
                    "(empty)"
                )

            lines.append("")

    total_time = (
        time.time()
        - started_all
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "END"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"TOTAL TIME: "
        f"{total_time:.2f}s"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    print()
    print("=" * 100)
    print(
        "PROMPT MODES TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()