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


from config import MODEL_CONFIG, RUNTIME_CONFIG  # noqa: E402
from inference import Assistant  # noqa: E402


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_quality_audit.txt"
)


MAX_NEW_TOKENS = 160


TESTS = [
    # ========================================================
    # BASIC SANITY
    # ========================================================

    {
        "category": "BASIC",
        "name": "2 + 2",
        "query": "Сколько будет 2 + 2?",
        "reference": (
            "Ожидается простой ответ: 4."
        ),
    },

    {
        "category": "BASIC",
        "name": "12 - 5",
        "query": (
            "У Маши было 12 яблок. "
            "Она отдала 5. "
            "Сколько яблок осталось?"
        ),
        "reference": (
            "Ожидается: 7 яблок."
        ),
    },

    {
        "category": "BASIC",
        "name": "17 * 23",
        "query": (
            "Сколько будет 17 умножить на 23?"
        ),
        "reference": (
            "Ожидается: 391."
        ),
    },

    {
        "category": "BASIC",
        "name": "Australia capital",
        "query": (
            "Какая столица Австралии?"
        ),
        "reference": (
            "Ожидается: Канберра."
        ),
    },

    {
        "category": "BASIC",
        "name": "Sky color",
        "query": (
            "Какого цвета обычно ясное "
            "дневное небо?"
        ),
        "reference": (
            "Ожидается: синее/голубое."
        ),
    },

    {
        "category": "BASIC",
        "name": "Water freezing",
        "query": (
            "При какой температуре "
            "замерзает чистая вода "
            "при нормальном атмосферном "
            "давлении?"
        ),
        "reference": (
            "Ожидается: около 0 °C."
        ),
    },

    # ========================================================
    # INSTRUCTION FOLLOWING
    # ========================================================

    {
        "category": "INSTRUCTION",
        "name": "One sentence",
        "query": (
            "Одним коротким предложением "
            "объясни, зачем нужен пароль."
        ),
        "reference": (
            "Должно быть одно короткое "
            "связное предложение о защите "
            "доступа."
        ),
    },

    # ========================================================
    # PYTHON
    # ========================================================

    {
        "category": "PYTHON",
        "name": "What is Python",
        "query": (
            "Что такое Python?"
        ),
        "reference": (
            "Язык программирования общего "
            "назначения; высокоуровневый, "
            "динамически типизированный."
        ),
    },

    {
        "category": "PYTHON",
        "name": "List vs tuple",
        "query": (
            "Чем список list отличается "
            "от tuple в Python?"
        ),
        "reference": (
            "Главное: list изменяемый, "
            "tuple обычно неизменяемый."
        ),
    },

    {
        "category": "PYTHON",
        "name": "is vs ==",
        "query": (
            "Когда в Python использовать "
            "is, а когда ==?"
        ),
        "reference": (
            "is проверяет идентичность "
            "объектов; == сравнивает значения."
        ),
    },

    {
        "category": "PYTHON",
        "name": "Mutable default",
        "query": (
            "Почему список нельзя безопасно "
            "использовать как значение "
            "аргумента функции по умолчанию "
            "в Python?"
        ),
        "reference": (
            "Default вычисляется один раз; "
            "один mutable объект повторно "
            "используется между вызовами. "
            "Обычно используют None."
        ),
    },

    {
        "category": "PYTHON",
        "name": "Closure",
        "query": (
            "Что такое замыкание "
            "в Python?"
        ),
        "reference": (
            "Функция сохраняет доступ "
            "к переменным из внешней "
            "лексической области."
        ),
    },

    {
        "category": "PYTHON",
        "name": "Negative indexing",
        "query": (
            "Что выведет этот код?\n\n"
            "numbers = [10, 20, 30]\n"
            "print(numbers[-1], numbers[-2])"
        ),
        "reference": (
            "Ожидается: 30 20."
        ),
    },

    # ========================================================
    # NETWORK / WEB
    # ========================================================

    {
        "category": "NETWORK",
        "name": "TCP vs UDP",
        "query": (
            "Объясни простыми словами "
            "разницу между TCP и UDP."
        ),
        "reference": (
            "TCP — соединение, надежная "
            "упорядоченная доставка; "
            "UDP — датаграммы без такой "
            "гарантии."
        ),
    },

    {
        "category": "NETWORK",
        "name": "HTTP vs HTTPS",
        "query": (
            "Чем HTTPS отличается от HTTP?"
        ),
        "reference": (
            "HTTPS — HTTP поверх TLS: "
            "шифрование, аутентификация "
            "сервера и защита целостности."
        ),
    },

    {
        "category": "NETWORK",
        "name": "IP vs DNS",
        "query": (
            "Чем IP-адрес отличается "
            "от DNS-имени?"
        ),
        "reference": (
            "IP — сетевой адрес; "
            "DNS-имя — человекочитаемое имя, "
            "которое обычно разрешается "
            "в IP-адрес."
        ),
    },

    {
        "category": "WEB",
        "name": "REST",
        "query": (
            "Что такое REST API?"
        ),
        "reference": (
            "REST — архитектурный стиль; "
            "ресурсы, единообразный интерфейс, "
            "stateless и другие ограничения. "
            "Не просто HTTP + JSON."
        ),
    },

    # ========================================================
    # DOCKER
    # ========================================================

    {
        "category": "DOCKER",
        "name": "What is Docker",
        "query": (
            "Что такое Docker?"
        ),
        "reference": (
            "Платформа/набор инструментов "
            "для сборки, распространения "
            "и запуска контейнеров."
        ),
    },

    {
        "category": "DOCKER",
        "name": "Container vs VM",
        "query": (
            "Чем Docker-контейнер "
            "отличается от виртуальной машины?"
        ),
        "reference": (
            "Контейнер обычно использует "
            "ядро host OS; VM содержит "
            "отдельную guest OS и ядро."
        ),
    },

    {
        "category": "DOCKER",
        "name": "Stop vs kill",
        "query": (
            "Чем docker stop отличается "
            "от docker kill?"
        ),
        "reference": (
            "stop сначала пытается завершить "
            "процесс корректно и ждёт; "
            "kill по умолчанию завершает "
            "немедленно."
        ),
    },

    # ========================================================
    # GIT
    # ========================================================

    {
        "category": "GIT",
        "name": "Fetch vs pull",
        "query": (
            "Чем git fetch отличается "
            "от git pull?"
        ),
        "reference": (
            "fetch получает изменения, "
            "но не интегрирует их автоматически; "
            "pull обычно fetch + integration."
        ),
    },

    {
        "category": "GIT",
        "name": "Merge vs rebase",
        "query": (
            "В чём разница между "
            "git merge и git rebase?"
        ),
        "reference": (
            "merge объединяет истории, "
            "обычно сохраняя ветвление; "
            "rebase переносит/пересоздаёт "
            "коммиты и переписывает историю."
        ),
    },

    # ========================================================
    # DATABASES
    # ========================================================

    {
        "category": "DATABASE",
        "name": "PostgreSQL",
        "query": (
            "Что такое PostgreSQL?"
        ),
        "reference": (
            "Свободная объектно-реляционная "
            "СУБД."
        ),
    },

    {
        "category": "DATABASE",
        "name": "SQL",
        "query": (
            "Что такое SQL и для чего "
            "он используется?"
        ),
        "reference": (
            "Декларативный язык для работы "
            "с реляционными БД: определение "
            "схем, запросы, изменение данных "
            "и т. д."
        ),
    },

    {
        "category": "DATABASE",
        "name": "Database index",
        "query": (
            "Зачем нужен индекс "
            "в базе данных и какой "
            "у него главный недостаток?"
        ),
        "reference": (
            "Ускоряет чтение/поиск/часть "
            "операций, но занимает место "
            "и увеличивает стоимость записей."
        ),
    },

    {
        "category": "DATABASE",
        "name": "COUNT",
        "query": (
            "Чем SQL COUNT(*) отличается "
            "от COUNT(column)?"
        ),
        "reference": (
            "COUNT(*) считает строки; "
            "COUNT(column) считает "
            "не-NULL значения столбца."
        ),
    },

    # ========================================================
    # PANDAS
    # ========================================================

    {
        "category": "PANDAS",
        "name": "Missing values",
        "query": (
            "Как в pandas заполнить "
            "пропущенные значения NaN?"
        ),
        "reference": (
            "Обычно fillna(), ffill(), bfill() "
            "или заполнение вычисленным "
            "значением — в зависимости "
            "от смысла данных."
        ),
    },
]


def compact(
    text: str,
    limit: int = 3000,
) -> str:
    text = str(
        text
    ).strip()

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + "..."
    )


def main():
    print()
    print("=" * 100)
    print(
        "NOVA — FINAL QUALITY AUDIT — NO RAG"
    )
    print("=" * 100)
    print()

    print(
        "Questions: "
        f"{len(TESTS)}"
    )

    print(
        "temperature = 0"
    )

    print(
        "top_k = OFF"
    )

    print(
        "top_p = OFF"
    )

    print(
        "repetition_penalty = 1"
    )

    print(
        "RAG = OFF"
    )

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
        "NOVA FINAL QUALITY AUDIT"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        f"Checkpoint: "
        f"{RUNTIME_CONFIG['checkpoint']}"
    )

    lines.append(
        f"Questions: {len(TESTS)}"
    )

    lines.append("")

    lines.append(
        "Generation settings:"
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

    lines.append(
        "  RAG = OFF"
    )

    lines.append("")

    category_times: dict[
        str,
        list[float],
    ] = {}

    empty_answers = 0

    total_started = time.time()

    for number, test in enumerate(
        TESTS,
        start=1,
    ):
        category = test[
            "category"
        ]

        print(
            f"[{number:02d}/"
            f"{len(TESTS):02d}] "
            f"[{category}] "
            f"{test['name']}"
        )

        started = time.time()

        (
            answer,
            contexts,
            rag_info,
        ) = assistant.generate(
            query=test[
                "query"
            ],
            max_new_tokens=(
                MAX_NEW_TOKENS
            ),
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=None,
            rag_enabled=False,
            rag_source="off",
            answer_mode="auto",
            top_k_docs=1,
            min_score=0.80,
            use_reranker=False,
            return_rag_info=True,
        )

        elapsed = (
            time.time()
            - started
        )

        category_times.setdefault(
            category,
            [],
        ).append(
            elapsed
        )

        answer = str(
            answer
        ).strip()

        if not answer:
            empty_answers += 1

        lines.append(
            "=" * 100
        )

        lines.append(
            f"QUESTION {number:02d}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"CATEGORY: {category}"
        )

        lines.append(
            f"NAME: {test['name']}"
        )

        lines.append("")

        lines.append(
            "QUERY:"
        )

        lines.append(
            test[
                "query"
            ]
        )

        lines.append("")

        lines.append(
            "REFERENCE / EXPECTED POINTS:"
        )

        lines.append(
            test[
                "reference"
            ]
        )

        lines.append("")

        lines.append(
            "NOVA ANSWER:"
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

        lines.append(
            f"ANSWER CHARS: "
            f"{len(answer)}"
        )

        lines.append(
            f"TIME: "
            f"{elapsed:.3f}s"
        )

        lines.append(
            "ROUTE: "
            f"{rag_info.get('route')}"
        )

        lines.append(
            "ANSWER SOURCE: "
            f"{rag_info.get('answer_source')}"
        )

        lines.append(
            "MODEL GENERATED: "
            f"{rag_info.get('model_generated')}"
        )

        lines.append(
            f"CONTEXT COUNT: "
            f"{len(contexts)}"
        )

        lines.append("")

        # Поле специально оставляем пустым.
        #
        # Оценку будем ставить после просмотра.
        lines.append(
            "MANUAL VERDICT: "
            "[NOT REVIEWED]"
        )

        lines.append("")

    total_time = (
        time.time()
        - total_started
    )

    # ========================================================
    # SUMMARY
    # ========================================================

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
        f"TOTAL QUESTIONS: "
        f"{len(TESTS)}"
    )

    lines.append(
        f"EMPTY ANSWERS: "
        f"{empty_answers}"
    )

    lines.append(
        f"TOTAL TIME: "
        f"{total_time:.2f}s"
    )

    lines.append("")

    lines.append(
        "AVERAGE TIME BY CATEGORY:"
    )

    for category in sorted(
        category_times
    ):
        values = (
            category_times[
                category
            ]
        )

        average = (
            sum(values)
            / len(values)
        )

        lines.append(
            f"  {category}: "
            f"{average:.3f}s"
        )

    lines.append("")

    lines.append(
        "Manual verdict options:"
    )

    lines.append(
        "  НОРМАЛЬНО"
    )

    lines.append(
        "  ЧАСТИЧНО"
    )

    lines.append(
        "  НЕВЕРНО"
    )

    lines.append(
        "  МУСОР"
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
    print("=" * 100)
    print(
        "FINAL QUALITY AUDIT FINISHED"
    )
    print("=" * 100)

    print(
        f"Questions: "
        f"{len(TESTS)}"
    )

    print(
        f"Empty answers: "
        f"{empty_answers}"
    )

    print(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    print()

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()