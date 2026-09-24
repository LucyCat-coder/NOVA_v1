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


from rag import Embedder, Retriever  # noqa: E402


INDEX_PATH = (
    PROJECT_ROOT
    / "rag"
    / "core_index"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "core_gate_test.txt"
)


# ------------------------------------------------------------
# POSITIVE
#
# Эти вопросы должны попадать в CORE.
# Формулировки специально НЕ совпадают дословно
# с canonical questions.
# ------------------------------------------------------------

POSITIVE_TESTS = [
    (
        "core_tcp_udp",
        "TCP надёжный, а UDP нет — в чём между ними реальная разница?",
    ),

    (
        "core_docker_vm",
        "Контейнер Docker и обычная VM используют систему одинаково или нет?",
    ),

    (
        "core_python_mutable_default",
        "Почему список в default-аргументе функции Python сохраняет изменения между вызовами?",
    ),

    (
        "core_git_merge_rebase",
        "Merge сохраняет историю, а rebase переписывает? Объясни различие.",
    ),

    (
        "core_pandas_missing",
        "Чем в pandas лучше заменить NaN в колонке?",
    ),

    (
        "core_postgresql",
        "Postgres — это язык, база данных или СУБД?",
    ),

    (
        "core_database_index",
        "Зачем таблице базы данных нужен индекс и какой у него минус?",
    ),

    (
        "core_rest_api",
        "REST — это просто HTTP с JSON или что-то другое?",
    ),

    (
        "core_http",
        "Как устроен обмен запросом и ответом в HTTP?",
    ),

    (
        "core_sql",
        "Для чего нужен язык SQL в реляционной базе?",
    ),

    (
        "core_python",
        "Python — что это за язык программирования?",
    ),

    (
        "core_docker",
        "Для чего вообще используется Docker?",
    ),
]


# ------------------------------------------------------------
# NEGATIVE / OUT OF DOMAIN
#
# CORE НЕ должен считать эти запросы своими.
# Здесь нет "правильного id".
# ------------------------------------------------------------

NEGATIVE_TESTS = [
    "Как приготовить борщ?",

    "Сколько будет 17 умножить на 23?",

    "Почему небо голубое?",

    "Как заменить колесо на автомобиле?",

    "Напиши короткое поздравление с днем рождения.",

    "Как работает квантовая запутанность?",

    "Какая столица Австралии?",

    "Как настроить nginx reverse proxy с двумя upstream серверами?",

    "Почему CUDA kernel получает out of memory?",

    "Как реализовать бинарное дерево поиска на C++?",

    "Что такое Kubernetes StatefulSet?",

    "Как работает OAuth 2.0 authorization code flow?",
]


def index_exists() -> bool:
    return (
        Path(
            str(INDEX_PATH)
            + ".faiss"
        ).exists()
        and Path(
            str(INDEX_PATH)
            + ".jsonl"
        ).exists()
    )


def get_result(
    retriever: Retriever,
    query: str,
):
    results = retriever.retrieve(
        query=query,
        top_k=3,
        min_score=None,
        use_reranker=False,
    )

    if not results:
        return {
            "top1_id": None,
            "top1_score": None,
            "top2_score": None,
            "margin": None,
            "results": [],
        }

    top1_score = float(
        results[0][1]
    )

    top2_score = (
        float(
            results[1][1]
        )
        if len(results) > 1
        else None
    )

    margin = (
        top1_score - top2_score
        if top2_score is not None
        else None
    )

    return {
        "top1_id": (
            results[0][2]
            .get("id")
        ),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "results": results,
    }


def append_result(
    lines: list[str],
    result: dict,
):
    lines.append(
        f"TOP1_ID: "
        f"{result['top1_id']}"
    )

    lines.append(
        "TOP1_SCORE: "
        + (
            f"{result['top1_score']:.6f}"
            if result["top1_score"]
            is not None
            else "None"
        )
    )

    lines.append(
        "TOP2_SCORE: "
        + (
            f"{result['top2_score']:.6f}"
            if result["top2_score"]
            is not None
            else "None"
        )
    )

    lines.append(
        "MARGIN: "
        + (
            f"{result['margin']:.6f}"
            if result["margin"]
            is not None
            else "None"
        )
    )

    lines.append("")

    for rank, (
        answer,
        score,
        metadata,
    ) in enumerate(
        result[
            "results"
        ],
        start=1,
    ):
        lines.append(
            f"#{rank} "
            f"score={score:.6f} "
            f"id={metadata.get('id')}"
        )

        lines.append(
            "question: "
            f"{metadata.get('question')}"
        )

        lines.append("")


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — CORE GATE TEST"
    )
    print("=" * 100)
    print()

    if not index_exists():
        raise FileNotFoundError(
            "core_index отсутствует."
        )

    print(
        "Loading embedder..."
    )

    embedder = Embedder(
        device="cuda",
        pooling_mode="masked_mean",
    )

    print(
        "Loading CORE index..."
    )

    retriever = Retriever(
        index_path=INDEX_PATH,
        embedder=embedder,
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA CORE GATE TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Goal:"
    )

    lines.append(
        "  measure positive paraphrases"
    )

    lines.append(
        "  measure unrelated/OOD queries"
    )

    lines.append(
        "  inspect TOP1 score + TOP1/TOP2 margin"
    )

    lines.append("")

    positive_scores = []
    positive_margins = []
    positive_correct = 0

    negative_scores = []
    negative_margins = []

    started_all = time.time()

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    lines.append(
        "=" * 100
    )

    lines.append(
        "POSITIVE PARAPHRASES"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    for number, (
        expected_id,
        query,
    ) in enumerate(
        POSITIVE_TESTS,
        start=1,
    ):
        print(
            f"[POS {number}/"
            f"{len(POSITIVE_TESTS)}] "
            f"{query}"
        )

        result = get_result(
            retriever,
            query,
        )

        correct = (
            result[
                "top1_id"
            ]
            == expected_id
        )

        if correct:
            positive_correct += 1

        if (
            result[
                "top1_score"
            ]
            is not None
        ):
            positive_scores.append(
                result[
                    "top1_score"
                ]
            )

        if (
            result[
                "margin"
            ]
            is not None
        ):
            positive_margins.append(
                result[
                    "margin"
                ]
            )

        lines.append(
            "-" * 100
        )

        lines.append(
            f"POSITIVE {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"EXPECTED: {expected_id}"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if correct
                else "FAIL"
            )
        )

        append_result(
            lines,
            result,
        )

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "NEGATIVE / OUT-OF-DOMAIN"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    for number, query in enumerate(
        NEGATIVE_TESTS,
        start=1,
    ):
        print(
            f"[NEG {number}/"
            f"{len(NEGATIVE_TESTS)}] "
            f"{query}"
        )

        result = get_result(
            retriever,
            query,
        )

        if (
            result[
                "top1_score"
            ]
            is not None
        ):
            negative_scores.append(
                result[
                    "top1_score"
                ]
            )

        if (
            result[
                "margin"
            ]
            is not None
        ):
            negative_margins.append(
                result[
                    "margin"
                ]
            )

        lines.append(
            "-" * 100
        )

        lines.append(
            f"NEGATIVE {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        append_result(
            lines,
            result,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_time = (
        time.time()
        - started_all
    )

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "SUMMARY"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        "Positive TOP1 accuracy: "
        f"{positive_correct}/"
        f"{len(POSITIVE_TESTS)}"
    )

    if positive_scores:
        lines.append(
            "Positive TOP1 score: "
            f"min={min(positive_scores):.6f}, "
            f"max={max(positive_scores):.6f}"
        )

    if positive_margins:
        lines.append(
            "Positive margin: "
            f"min={min(positive_margins):.6f}, "
            f"max={max(positive_margins):.6f}"
        )

    if negative_scores:
        lines.append(
            "Negative TOP1 score: "
            f"min={min(negative_scores):.6f}, "
            f"max={max(negative_scores):.6f}"
        )

    if negative_margins:
        lines.append(
            "Negative margin: "
            f"min={min(negative_margins):.6f}, "
            f"max={max(negative_margins):.6f}"
        )

    lines.append(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    lines.append(
        "=" * 100
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
        "Positive TOP1: "
        f"{positive_correct}/"
        f"{len(POSITIVE_TESTS)}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "CORE GATE TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()