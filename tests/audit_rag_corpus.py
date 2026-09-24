from __future__ import annotations

import json
import re
import sys
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


INDEX_JSONL = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rag_corpus_audit.txt"
)


QUERIES = [
    "Что такое TCP и чем он отличается от UDP?",

    "Чем Docker-контейнер отличается от виртуальной машины?",

    "Почему нельзя использовать изменяемый объект как значение аргумента по умолчанию в Python?",

    "В чём разница между git merge и git rebase?",

    "Как заполнить пропущенные значения в pandas DataFrame?",

    "Что такое PostgreSQL?",

    "Как работает индекс в базе данных?",

    "Что такое REST API?",
]


STOPWORDS = {
    "что",
    "такое",
    "как",
    "какой",
    "какая",
    "какие",
    "каким",
    "каким образом",
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
}


def normalize(
    text: str,
) -> str:
    text = (
        text
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


def tokens(
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


def lexical_score(
    query: str,
    candidate: str,
) -> float:
    query_norm = normalize(
        query
    )

    candidate_norm = normalize(
        candidate
    )

    query_tokens = tokens(
        query
    )

    candidate_tokens = tokens(
        candidate
    )

    if not query_tokens:
        token_score = 0.0

    else:
        intersection = len(
            query_tokens
            & candidate_tokens
        )

        token_score = (
            intersection
            / len(query_tokens)
        )

    sequence_score = (
        SequenceMatcher(
            None,
            query_norm,
            candidate_norm,
        )
        .ratio()
    )

    containment_bonus = 0.0

    if query_norm in candidate_norm:
        containment_bonus = 0.25

    score = (
        token_score * 0.70
        + sequence_score * 0.30
        + containment_bonus
    )

    return score


def load_unique_questions():
    questions = {}

    with INDEX_JSONL.open(
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
                "?",
            )

            key = (
                str(q_idx),
                question,
            )

            if key not in questions:
                questions[key] = {
                    "question": question,
                    "q_idx": q_idx,
                    "chunks": [],
                }

            questions[
                key
            ]["chunks"].append(
                {
                    "chunk_id": meta.get(
                        "chunk_id",
                        "?",
                    ),
                    "text": record.get(
                        "text",
                        "",
                    ),
                }
            )

    return list(
        questions.values()
    )


def compact(
    text: str,
    limit: int = 650,
) -> str:
    text = (
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )

    while "  " in text:
        text = text.replace(
            "  ",
            " ",
        )

    if len(text) > limit:
        return (
            text[:limit]
            + "..."
        )

    return text


def main():
    if not INDEX_JSONL.exists():
        raise FileNotFoundError(
            INDEX_JSONL
        )

    print()
    print("=" * 100)
    print("NOVA — RAG CORPUS AUDIT")
    print("=" * 100)

    print()
    print(
        f"Source: {INDEX_JSONL}"
    )

    print(
        "Reading unique source questions..."
    )

    questions = (
        load_unique_questions()
    )

    print(
        "Unique questions: "
        f"{len(questions):,}"
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA RAG CORPUS AUDIT"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        f"Source: {INDEX_JSONL}"
    )

    lines.append(
        "Unique source questions: "
        f"{len(questions)}"
    )

    lines.append("")

    for number, query in enumerate(
        QUERIES,
        start=1,
    ):
        print(
            f"[{number}/{len(QUERIES)}] "
            f"{query}"
        )

        scored = []

        for item in questions:
            score = lexical_score(
                query,
                item["question"],
            )

            scored.append(
                (
                    score,
                    item,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        top_results = (
            scored[:10]
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUERY {number}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"User query: {query}"
        )

        lines.append("")

        lines.append(
            "TOP 10 SOURCE QUESTIONS "
            "(LEXICAL SEARCH)"
        )

        lines.append(
            "-" * 100
        )

        for rank, (
            score,
            item,
        ) in enumerate(
            top_results,
            start=1,
        ):
            lines.append(
                f"#{rank} "
                f"score={score:.6f}"
            )

            lines.append(
                f"q_idx={item['q_idx']}"
            )

            lines.append(
                "source question: "
                f"{item['question']}"
            )

            lines.append(
                "chunks: "
                f"{len(item['chunks'])}"
            )

            if item[
                "chunks"
            ]:
                first_chunk = (
                    item[
                        "chunks"
                    ][0]["text"]
                )

                lines.append(
                    "first chunk:"
                )

                lines.append(
                    compact(
                        first_chunk
                    )
                )

            lines.append("")

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print("CORPUS AUDIT FINISHED")
    print("=" * 100)


if __name__ == "__main__":
    main()