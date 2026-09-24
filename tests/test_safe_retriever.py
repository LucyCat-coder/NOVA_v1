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


from rag import SafeRetriever  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "safe_retriever_test.txt"
)


CORE_TESTS = [
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


AUTO_REJECT_TESTS = [
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


TECHNICAL_TESTS = [
    (
        2,
        "Когда в Python использовать is, а когда ==?",
    ),
    (
        39,
        "Что такое замыкание в Python?",
    ),
    (
        267,
        "Чем docker stop отличается от docker kill?",
    ),
    (
        149,
        "Чем git fetch отличается от git pull?",
    ),
]


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — SAFE RETRIEVER TEST"
    )
    print("=" * 100)
    print()

    retriever = SafeRetriever(
        device="cuda",
        core_threshold=0.83,
    )

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "NOVA SAFE RETRIEVER TEST"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "AUTO = CORE only"
    )

    lines.append(
        "CORE threshold = 0.83"
    )

    lines.append(
        "TECHNICAL/BROAD = manual only"
    )

    lines.append("")

    core_pass = 0
    reject_pass = 0
    technical_pass = 0

    started = time.time()

    # --------------------------------------------------------
    # AUTO CORE ACCEPT
    # --------------------------------------------------------

    lines.append(
        "=" * 100
    )

    lines.append(
        "AUTO — CORE ACCEPT"
    )

    lines.append(
        "=" * 100
    )

    for number, (
        expected_id,
        query,
    ) in enumerate(
        CORE_TESTS,
        start=1,
    ):
        result = (
            retriever.retrieve_with_info(
                query=query,
                mode="auto",
            )
        )

        results = result[
            "results"
        ]

        actual_id = (
            results[0][2].get(
                "id"
            )
            if results
            else None
        )

        ok = (
            result[
                "accepted"
            ]
            and result[
                "route"
            ]
            == "core"
            and actual_id
            == expected_id
        )

        if ok:
            core_pass += 1

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"CORE {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"EXPECTED: {expected_id}"
        )

        lines.append(
            f"ACTUAL: {actual_id}"
        )

        lines.append(
            f"SCORE: "
            f"{result['top1_score']}"
        )

        lines.append(
            f"ROUTE: "
            f"{result['route']}"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

    # --------------------------------------------------------
    # AUTO REJECT
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "AUTO — REJECT"
    )

    lines.append(
        "=" * 100
    )

    for number, query in enumerate(
        AUTO_REJECT_TESTS,
        start=1,
    ):
        result = (
            retriever.retrieve_with_info(
                query=query,
                mode="auto",
            )
        )

        ok = (
            not result[
                "accepted"
            ]
            and result[
                "route"
            ]
            == "none"
            and not result[
                "results"
            ]
        )

        if ok:
            reject_pass += 1

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"REJECT {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"CORE SCORE: "
            f"{result['top1_score']}"
        )

        lines.append(
            f"ROUTE: "
            f"{result['route']}"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

    # --------------------------------------------------------
    # MANUAL TECHNICAL
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 100
    )

    lines.append(
        "MANUAL TECHNICAL"
    )

    lines.append(
        "=" * 100
    )

    for number, (
        expected_line,
        query,
    ) in enumerate(
        TECHNICAL_TESTS,
        start=1,
    ):
        result = (
            retriever.retrieve_with_info(
                query=query,
                mode="technical",
                top_k=1,
            )
        )

        results = result[
            "results"
        ]

        actual_line = (
            results[0][2].get(
                "line"
            )
            if results
            else None
        )

        ok = (
            actual_line
            == expected_line
        )

        if ok:
            technical_pass += 1

        lines.append("")
        lines.append(
            "-" * 100
        )

        lines.append(
            f"TECHNICAL {number}"
        )

        lines.append(
            f"QUERY: {query}"
        )

        lines.append(
            f"EXPECTED LINE: "
            f"{expected_line}"
        )

        lines.append(
            f"ACTUAL LINE: "
            f"{actual_line}"
        )

        lines.append(
            f"SCORE: "
            f"{result['top1_score']}"
        )

        lines.append(
            "RESULT: "
            + (
                "PASS"
                if ok
                else "FAIL"
            )
        )

    total_time = (
        time.time()
        - started
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
        f"AUTO CORE: "
        f"{core_pass}/"
        f"{len(CORE_TESTS)}"
    )

    lines.append(
        f"AUTO REJECT: "
        f"{reject_pass}/"
        f"{len(AUTO_REJECT_TESTS)}"
    )

    lines.append(
        f"MANUAL TECHNICAL: "
        f"{technical_pass}/"
        f"{len(TECHNICAL_TESTS)}"
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

    print(
        f"AUTO CORE: "
        f"{core_pass}/"
        f"{len(CORE_TESTS)}"
    )

    print(
        f"AUTO REJECT: "
        f"{reject_pass}/"
        f"{len(AUTO_REJECT_TESTS)}"
    )

    print(
        f"MANUAL TECHNICAL: "
        f"{technical_pass}/"
        f"{len(TECHNICAL_TESTS)}"
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    print()
    print("=" * 100)
    print(
        "SAFE RETRIEVER TEST FINISHED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()