from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
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


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "nova_technical_v1_1"
    / "nova_technical_v1_1_prepared.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "technical_dataset_audit.txt"
)


AUDIT_QUERIES = [
    "Что такое TCP и чем он отличается от UDP?",

    "Чем Docker-контейнер отличается от виртуальной машины?",

    "Почему нельзя использовать изменяемый объект как значение аргумента по умолчанию в Python?",

    "В чём разница между git merge и git rebase?",

    "Как заполнить пропущенные значения в pandas DataFrame?",

    "Что такое PostgreSQL?",

    "Как работает индекс в базе данных?",

    "Что такое REST API?",

    "Что такое HTTP?",

    "Что такое SQL?",

    "Что такое Python?",

    "Что такое Docker?",
]


STOPWORDS = {
    "что",
    "такое",
    "как",
    "какой",
    "какая",
    "какие",
    "каким",
    "чем",
    "он",
    "она",
    "они",
    "это",
    "и",
    "или",
    "в",
    "во",
    "на",
    "для",
    "из",
    "с",
    "со",
    "по",
    "к",
    "у",
    "при",
    "если",
    "можно",
    "нельзя",
    "почему",
    "между",
    "от",
    "ли",
    "же",
    "бы",
}


SUSPICIOUS_ANSWER_PATTERNS = [
    r"я\s+модель\s+ии",
    r"я\s+языковая\s+модель",
    r"как\s+искусственный\s+интеллект",
    r"у\s+меня\s+нет\s+доступа",
    r"не\s+могу\s+просматривать",
    r"не\s+могу\s+видеть",
    r"извините.{0,40}не\s+могу",
    r"ошибки:\s*похоже",
    r"уточняющий\s+вопрос:",
    r"добрый\s+день.{0,80}рад\s+помочь",
]


def normalize(
    text: str,
) -> str:
    text = (
        str(text)
        .lower()
        .replace("ё", "е")
    )

    text = re.sub(
        r"[^a-zа-я0-9+#._-]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def token_set(
    text: str,
) -> set[str]:
    result = set()

    for token in normalize(
        text
    ).split():
        if token in STOPWORDS:
            continue

        if len(token) <= 1:
            continue

        result.add(
            token
        )

    return result


def similarity(
    query: str,
    candidate: str,
) -> float:
    q_norm = normalize(
        query
    )

    c_norm = normalize(
        candidate
    )

    q_tokens = token_set(
        query
    )

    c_tokens = token_set(
        candidate
    )

    if q_tokens:
        coverage = (
            len(
                q_tokens
                & c_tokens
            )
            / len(q_tokens)
        )
    else:
        coverage = 0.0

    sequence = (
        SequenceMatcher(
            None,
            q_norm,
            c_norm,
        )
        .ratio()
    )

    containment = 0.0

    if (
        q_norm
        and q_norm in c_norm
    ):
        containment = 0.25

    return (
        coverage * 0.70
        + sequence * 0.30
        + containment
    )


def compact(
    text: str,
    limit: int = 1000,
) -> str:
    text = (
        str(text)
        .replace("\r", " ")
        .strip()
    )

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def extract_from_messages(
    messages,
):
    if not isinstance(
        messages,
        list,
    ):
        return None

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
            extracted = []

            for item in content:
                if isinstance(
                    item,
                    str,
                ):
                    extracted.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):
                    text = item.get(
                        "text",
                        "",
                    )

                    if text:
                        extracted.append(
                            str(text)
                        )

            content = "\n".join(
                extracted
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
        }:
            assistant_parts.append(
                content
            )

    if (
        user_parts
        and assistant_parts
    ):
        return (
            "\n".join(
                user_parts
            ),
            "\n".join(
                assistant_parts
            ),
        )

    return None


def extract_qa(
    record: dict,
):
    # --------------------------------------------------------
    # OpenAI-style messages
    # --------------------------------------------------------

    if "messages" in record:
        result = extract_from_messages(
            record["messages"]
        )

        if result is not None:
            return (
                result[0],
                result[1],
                "messages",
            )

    # --------------------------------------------------------
    # conversations-style
    # --------------------------------------------------------

    if "conversations" in record:
        conversations = (
            record[
                "conversations"
            ]
        )

        if isinstance(
            conversations,
            list,
        ):
            user_parts = []
            assistant_parts = []

            for item in conversations:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                role = str(
                    item.get(
                        "from",
                        item.get(
                            "role",
                            "",
                        ),
                    )
                ).strip().lower()

                value = str(
                    item.get(
                        "value",
                        item.get(
                            "content",
                            "",
                        ),
                    )
                ).strip()

                if not value:
                    continue

                if role in {
                    "human",
                    "user",
                }:
                    user_parts.append(
                        value
                    )

                elif role in {
                    "assistant",
                    "gpt",
                    "bot",
                }:
                    assistant_parts.append(
                        value
                    )

            if (
                user_parts
                and assistant_parts
            ):
                return (
                    "\n".join(
                        user_parts
                    ),
                    "\n".join(
                        assistant_parts
                    ),
                    "conversations",
                )

    # --------------------------------------------------------
    # Common direct pairs
    # --------------------------------------------------------

    candidate_pairs = [
        (
            "question",
            "answer",
        ),
        (
            "prompt",
            "response",
        ),
        (
            "prompt",
            "completion",
        ),
        (
            "instruction",
            "output",
        ),
        (
            "input",
            "output",
        ),
        (
            "user",
            "assistant",
        ),
        (
            "query",
            "response",
        ),
    ]

    for q_key, a_key in candidate_pairs:
        if (
            q_key in record
            and a_key in record
        ):
            question = str(
                record[
                    q_key
                ]
            ).strip()

            answer = str(
                record[
                    a_key
                ]
            ).strip()

            if (
                question
                or answer
            ):
                return (
                    question,
                    answer,
                    (
                        f"{q_key}+"
                        f"{a_key}"
                    ),
                )

    return (
        "",
        "",
        "unknown",
    )


def suspicious_patterns(
    answer: str,
) -> list[str]:
    normalized = (
        answer
        .lower()
        .replace("ё", "е")
    )

    found = []

    for pattern in (
        SUSPICIOUS_ANSWER_PATTERNS
    ):
        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
            | re.DOTALL,
        ):
            found.append(
                pattern
            )

    return found


def load_dataset():
    records = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            raw_line = line.strip()

            if not raw_line:
                continue

            try:
                raw = json.loads(
                    raw_line
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Некорректный JSON "
                    f"в строке {line_number}"
                ) from exc

            if not isinstance(
                raw,
                dict,
            ):
                records.append(
                    {
                        "line": line_number,
                        "schema": (
                            "non-dict"
                        ),
                        "question": "",
                        "answer": "",
                        "raw": raw,
                    }
                )

                continue

            (
                question,
                answer,
                schema,
            ) = extract_qa(
                raw
            )

            records.append(
                {
                    "line": (
                        line_number
                    ),
                    "schema": schema,
                    "question": (
                        question
                    ),
                    "answer": (
                        answer
                    ),
                    "raw": raw,
                }
            )

    return records


def percentile(
    values: list[int],
    fraction: float,
) -> int:
    if not values:
        return 0

    sorted_values = sorted(
        values
    )

    index = int(
        round(
            (
                len(
                    sorted_values
                )
                - 1
            )
            * fraction
        )
    )

    return sorted_values[
        index
    ]


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            DATASET_PATH
        )

    print()
    print("=" * 100)
    print(
        "NOVA — TECHNICAL DATASET AUDIT"
    )
    print("=" * 100)
    print()

    print(
        f"Source: {DATASET_PATH}"
    )

    print(
        "Reading dataset..."
    )

    records = load_dataset()

    total = len(
        records
    )

    print(
        f"Records: {total:,}"
    )

    schema_counts = Counter(
        item[
            "schema"
        ]
        for item in records
    )

    extracted = [
        item
        for item in records
        if (
            item[
                "question"
            ].strip()
            and item[
                "answer"
            ].strip()
        )
    ]

    empty_question = [
        item
        for item in records
        if not item[
            "question"
        ].strip()
    ]

    empty_answer = [
        item
        for item in records
        if not item[
            "answer"
        ].strip()
    ]

    question_counter = Counter(
        normalize(
            item[
                "question"
            ]
        )
        for item in extracted
        if normalize(
            item[
                "question"
            ]
        )
    )

    answer_counter = Counter(
        normalize(
            item[
                "answer"
            ]
        )
        for item in extracted
        if normalize(
            item[
                "answer"
            ]
        )
    )

    duplicate_questions = {
        key: count
        for key, count
        in question_counter.items()
        if count > 1
    }

    duplicate_answers = {
        key: count
        for key, count
        in answer_counter.items()
        if count > 1
    }

    question_lengths = [
        len(
            item[
                "question"
            ]
        )
        for item in extracted
    ]

    answer_lengths = [
        len(
            item[
                "answer"
            ]
        )
        for item in extracted
    ]

    short_answers = [
        item
        for item in extracted
        if len(
            item[
                "answer"
            ].strip()
        ) < 80
    ]

    very_short_answers = [
        item
        for item in extracted
        if len(
            item[
                "answer"
            ].strip()
        ) < 30
    ]

    very_long_answers = [
        item
        for item in extracted
        if len(
            item[
                "answer"
            ]
        ) > 5000
    ]

    suspicious = []

    for item in extracted:
        patterns = (
            suspicious_patterns(
                item[
                    "answer"
                ]
            )
        )

        if patterns:
            suspicious.append(
                (
                    item,
                    patterns,
                )
            )

    report = []

    def add(
        text: str = "",
    ):
        report.append(
            text
        )

    add(
        "=" * 100
    )

    add(
        "NOVA TECHNICAL DATASET AUDIT"
    )

    add(
        "=" * 100
    )

    add()

    add(
        f"Source: {DATASET_PATH}"
    )

    add(
        f"Total records: {total}"
    )

    add(
        "Successfully extracted Q/A: "
        f"{len(extracted)}"
    )

    add(
        "Empty/unrecognized question: "
        f"{len(empty_question)}"
    )

    add(
        "Empty/unrecognized answer: "
        f"{len(empty_answer)}"
    )

    add()

    add(
        "SCHEMAS"
    )

    add(
        "-" * 100
    )

    for schema, count in (
        schema_counts
        .most_common()
    ):
        add(
            f"{schema}: {count}"
        )

    add()

    add(
        "DUPLICATES"
    )

    add(
        "-" * 100
    )

    add(
        "Duplicated normalized questions: "
        f"{len(duplicate_questions)}"
    )

    add(
        "Duplicated normalized answers: "
        f"{len(duplicate_answers)}"
    )

    add()

    add(
        "LENGTH STATISTICS"
    )

    add(
        "-" * 100
    )

    if question_lengths:
        add(
            "Question chars: "
            f"min={min(question_lengths)}, "
            f"p50={percentile(question_lengths, 0.50)}, "
            f"p90={percentile(question_lengths, 0.90)}, "
            f"max={max(question_lengths)}"
        )

    if answer_lengths:
        add(
            "Answer chars: "
            f"min={min(answer_lengths)}, "
            f"p50={percentile(answer_lengths, 0.50)}, "
            f"p90={percentile(answer_lengths, 0.90)}, "
            f"max={max(answer_lengths)}"
        )

    add(
        "Answers < 80 chars: "
        f"{len(short_answers)}"
    )

    add(
        "Answers < 30 chars: "
        f"{len(very_short_answers)}"
    )

    add(
        "Answers > 5000 chars: "
        f"{len(very_long_answers)}"
    )

    add()

    add(
        "HEURISTIC QUALITY FLAGS"
    )

    add(
        "-" * 100
    )

    add(
        "Answers matching suspicious "
        "assistant/meta patterns: "
        f"{len(suspicious)}"
    )

    for item, patterns in (
        suspicious[:20]
    ):
        add()

        add(
            f"line={item['line']}"
        )

        add(
            "question: "
            f"{compact(item['question'], 300)}"
        )

        add(
            "patterns: "
            f"{patterns}"
        )

        add(
            "answer:"
        )

        add(
            compact(
                item[
                    "answer"
                ],
                600,
            )
        )

    add()
    add()

    add(
        "=" * 100
    )

    add(
        "TOPICAL COVERAGE"
    )

    add(
        "=" * 100
    )

    add()

    for number, query in enumerate(
        AUDIT_QUERIES,
        start=1,
    ):
        scored = []

        for item in extracted:
            score = similarity(
                query,
                item[
                    "question"
                ],
            )

            scored.append(
                (
                    score,
                    item,
                )
            )

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        top = scored[:8]

        add(
            "=" * 100
        )

        add(
            f"QUERY {number}"
        )

        add(
            "=" * 100
        )

        add(
            f"Audit query: {query}"
        )

        add()

        for rank, (
            score,
            item,
        ) in enumerate(
            top,
            start=1,
        ):
            add(
                f"#{rank} "
                f"score={score:.6f} "
                f"line={item['line']}"
            )

            add(
                "QUESTION:"
            )

            add(
                compact(
                    item[
                        "question"
                    ],
                    500,
                )
            )

            add()

            add(
                "ANSWER:"
            )

            add(
                compact(
                    item[
                        "answer"
                    ],
                    1000,
                )
            )

            add()

    add(
        "=" * 100
    )

    add(
        "END OF AUDIT"
    )

    add(
        "=" * 100
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Extracted Q/A: "
        f"{len(extracted):,}/"
        f"{total:,}"
    )

    print(
        "Duplicated questions: "
        f"{len(duplicate_questions):,}"
    )

    print(
        "Duplicated answers: "
        f"{len(duplicate_answers):,}"
    )

    print(
        "Suspicious answers: "
        f"{len(suspicious):,}"
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "TECHNICAL DATASET AUDIT FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()