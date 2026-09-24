from __future__ import annotations

import threading
import time
from pathlib import Path

import gradio as gr

from config import MODEL_CONFIG, RUNTIME_CONFIG
from inference import Assistant


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# UI MODES
# ============================================================

MODE_AUTO = "Auto — безопасный"
MODE_TECHNICAL = "Technical — проверенные technical Q/A"
MODE_BROAD = "Broad — экспериментальный"
MODE_OFF = "No RAG — только модель"


MODE_CONFIG = {
    MODE_AUTO: {
        "rag_enabled": True,
        "rag_source": "auto",
        "answer_mode": "auto",
    },

    MODE_TECHNICAL: {
        "rag_enabled": True,
        "rag_source": "technical",
        "answer_mode": "auto",
    },

    MODE_BROAD: {
        "rag_enabled": True,
        "rag_source": "broad",
        "answer_mode": "generate",
    },

    MODE_OFF: {
        "rag_enabled": False,
        "rag_source": "off",
        "answer_mode": "auto",
    },
}


# ============================================================
# LAZY ASSISTANT
# ============================================================

_assistant: Assistant | None = None

_assistant_lock = threading.Lock()

_generation_lock = threading.Lock()


def get_assistant() -> Assistant:
    """
    Модель загружается только при первом запросе.

    Благодаря этому Gradio-интерфейс может
    открыться до загрузки checkpoint / E5.
    """

    global _assistant

    if _assistant is not None:
        return _assistant

    with _assistant_lock:
        if _assistant is not None:
            return _assistant

        print()
        print("=" * 80)
        print("Loading NOVA for web interface...")
        print("=" * 80)

        _assistant = Assistant(
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

        print()
        print("NOVA web assistant ready.")
        print()

        return _assistant


# ============================================================
# HELPERS
# ============================================================

def format_score(
    value,
) -> str:
    if value is None:
        return "—"

    try:
        return f"{float(value):.6f}"

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def compact_text(
    text: str,
    limit: int = 800,
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


def make_diagnostics(
    *,
    selected_mode: str,
    elapsed: float,
    contexts: list[str],
    rag_info: dict,
) -> str:
    route = rag_info.get(
        "route",
        "off",
    )

    answer_source = rag_info.get(
        "answer_source",
        "model",
    )

    top1_score = rag_info.get(
        "top1_score",
        None,
    )

    threshold = rag_info.get(
        "threshold",
        None,
    )

    direct_answer = rag_info.get(
        "direct_answer",
        False,
    )

    model_generated = rag_info.get(
        "model_generated",
        False,
    )

    results = rag_info.get(
        "results",
        [],
    )

    metadata = {}

    if results:
        metadata = (
            results[0][2]
            or {}
        )

    source_question = metadata.get(
        "question",
        None,
    )

    source_type = metadata.get(
        "source_type",
        None,
    )

    source_id = metadata.get(
        "id",
        None,
    )

    source_line = metadata.get(
        "line",
        None,
    )

    lines = [
        "### Диагностика",
        "",
        f"- UI mode: `{selected_mode}`",
        f"- Route: `{route}`",
        (
            "- Источник ответа: "
            f"`{answer_source}`"
        ),
        (
            "- Direct retrieval: "
            f"`{direct_answer}`"
        ),
        (
            "- Model generated: "
            f"`{model_generated}`"
        ),
        (
            "- Retrieval score: "
            f"`{format_score(top1_score)}`"
        ),
    ]

    if threshold is not None:
        lines.append(
            "- Threshold: "
            f"`{format_score(threshold)}`"
        )

    lines.append(
        f"- Время: `{elapsed:.3f}s`"
    )

    if source_type:
        lines.append(
            "- Source type: "
            f"`{source_type}`"
        )

    if source_id:
        lines.append(
            "- ID: "
            f"`{source_id}`"
        )

    if source_line is not None:
        lines.append(
            "- Dataset line: "
            f"`{source_line}`"
        )

    if source_question:
        lines.extend(
            [
                "",
                "**Найденный исходный вопрос:**",
                "",
                compact_text(
                    source_question,
                    500,
                ),
            ]
        )

    # --------------------------------------------------------
    # HUMAN-READABLE ROUTE EXPLANATION
    # --------------------------------------------------------

    lines.append("")

    if (
        route == "core"
        and answer_source
        == "retrieval"
    ):
        lines.append(
            "✅ CORE принят. "
            "Ответ возвращён напрямую "
            "из проверенной базы; "
            "FINAL checkpoint его не переписывал."
        )

    elif (
        route == "technical"
        and answer_source
        == "retrieval"
    ):
        lines.append(
            "✅ Использован Technical Q/A. "
            "Ответ возвращён напрямую "
            "из technical dataset."
        )

    elif route == "none":
        lines.append(
            "⚪ CORE не прошёл confidence gate. "
            "Контекст не подмешивался; "
            "ответ сгенерирован FINAL NOVA."
        )

    elif route == "broad":
        lines.append(
            "⚠️ Использован BROAD V2. "
            "Это экспериментальный источник: "
            "контекст может быть шумным, "
            "а FINAL NOVA может интерпретировать "
            "его неточно."
        )

    elif route == "off":
        lines.append(
            "⚪ RAG отключён. "
            "Ответ полностью сгенерирован "
            "FINAL NOVA."
        )

    # --------------------------------------------------------
    # CONTEXT PREVIEW
    # --------------------------------------------------------

    if contexts:
        lines.extend(
            [
                "",
                "**Контекст / retrieved answer:**",
                "",
                "```text",
                compact_text(
                    contexts[0],
                    1000,
                ),
                "```",
            ]
        )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN QUERY HANDLER
# ============================================================

def answer_query(
    query: str,
    selected_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
):
    query = str(
        query
    ).strip()

    if not query:
        return (
            "Введите вопрос.",
            (
                "### Диагностика\n\n"
                "Запрос пуст."
            ),
        )

    if (
        selected_mode
        not in MODE_CONFIG
    ):
        return (
            "Неизвестный режим.",
            (
                "### Диагностика\n\n"
                f"Unknown mode: "
                f"`{selected_mode}`"
            ),
        )

    mode = MODE_CONFIG[
        selected_mode
    ]

    try:
        assistant = (
            get_assistant()
        )

        started = time.time()

        # Один локальный GPU.
        #
        # Не запускаем одновременно
        # две генерации / retrieval операции.
        with _generation_lock:
            (
                answer,
                contexts,
                rag_info,
            ) = assistant.generate(
                query=query,
                max_new_tokens=int(
                    max_new_tokens
                ),
                temperature=float(
                    temperature
                ),
                top_k=(
                    int(top_k)
                    if int(top_k) > 0
                    else None
                ),
                top_p=float(
                    top_p
                ),
                repetition_penalty=float(
                    repetition_penalty
                ),
                seed=None,
                rag_enabled=(
                    mode[
                        "rag_enabled"
                    ]
                ),
                rag_source=(
                    mode[
                        "rag_source"
                    ]
                ),
                answer_mode=(
                    mode[
                        "answer_mode"
                    ]
                ),
                rag_mode=(
                    RUNTIME_CONFIG[
                        "rag_mode"
                    ]
                ),
                top_k_docs=(
                    RUNTIME_CONFIG[
                        "top_k_docs"
                    ]
                ),
                min_score=(
                    RUNTIME_CONFIG[
                        "min_score"
                    ]
                ),
                use_reranker=False,
                return_rag_info=True,
            )

        elapsed = (
            time.time()
            - started
        )

        answer = str(
            answer
        ).strip()

        if not answer:
            answer = (
                "NOVA вернула пустой ответ."
            )

        diagnostics = (
            make_diagnostics(
                selected_mode=(
                    selected_mode
                ),
                elapsed=elapsed,
                contexts=contexts,
                rag_info=rag_info,
            )
        )

        return (
            answer,
            diagnostics,
        )

    except Exception as exc:
        print()
        print(
            "[APP ERROR] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return (
            (
                "Произошла ошибка при "
                "обработке запроса."
            ),
            (
                "### Диагностика ошибки\n\n"
                f"- Type: "
                f"`{type(exc).__name__}`\n"
                f"- Message: "
                f"`{exc}`\n\n"
                "Полный traceback смотри "
                "в PowerShell."
            ),
        )


# ============================================================
# CLEAR
# ============================================================

def clear_interface():
    return (
        "",
        "",
        (
            "### Диагностика\n\n"
            "Новый запрос."
        ),
    )


# ============================================================
# GRADIO UI
# ============================================================

def build_interface():
    with gr.Blocks(
        title="NOVA"
    ) as demo:
        gr.Markdown(
            """
# NOVA

Локальная версия NOVA.

**Рекомендуемый режим — Auto.**

В Auto сначала проверяется маленькая
курируемая база CORE. Если совпадение
достаточно уверенное, проверенный ответ
возвращается напрямую. Если CORE не уверен,
запрос передаётся FINAL NOVA без RAG.

Каждый запрос пока обрабатывается
**независимо**: история прошлых сообщений
не передаётся модели.
"""
        )

        with gr.Row():
            selected_mode = gr.Radio(
                choices=[
                    MODE_AUTO,
                    MODE_TECHNICAL,
                    MODE_BROAD,
                    MODE_OFF,
                ],
                value=MODE_AUTO,
                label="Режим",
            )

        query = gr.Textbox(
            label="Вопрос",
            placeholder=(
                "Например: "
                "Что такое PostgreSQL?"
            ),
            lines=4,
        )

        with gr.Row():
            submit_button = gr.Button(
                "Отправить",
                variant="primary",
            )

            clear_button = gr.Button(
                "Очистить",
            )

        gr.Markdown(
            """
### Что означают режимы

**Auto — безопасный**  
CORE → прямой проверенный ответ.  
Если CORE не уверен → FINAL NOVA.

**Technical**  
Ищет в `nova_technical_v1_1` и возвращает
найденный technical answer напрямую.
Это ручной режим для технических вопросов.

**Broad — экспериментальный**  
Ищет в большом V2-корпусе и передаёт
контекст FINAL NOVA. Корпус шумный,
а grounding модели пока слабый.

**No RAG**  
Только FINAL NOVA.
"""
        )

        with gr.Accordion(
            "Параметры генерации",
            open=False,
        ):
            max_new_tokens = gr.Slider(
                minimum=32,
                maximum=300,
                value=(
                    RUNTIME_CONFIG[
                        "max_new_tokens"
                    ]
                ),
                step=1,
                label="Max new tokens",
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=(
                    RUNTIME_CONFIG[
                        "temperature"
                    ]
                ),
                step=0.05,
                label="Temperature",
            )

            top_k = gr.Slider(
                minimum=0,
                maximum=100,
                value=(
                    RUNTIME_CONFIG[
                        "top_k"
                    ]
                ),
                step=1,
                label=(
                    "Top-k "
                    "(0 = выключено)"
                ),
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=(
                    RUNTIME_CONFIG[
                        "top_p"
                    ]
                ),
                step=0.05,
                label=(
                    "Top-p "
                    "(1.0 = выключено)"
                ),
            )

            repetition_penalty = (
                gr.Slider(
                    minimum=1.0,
                    maximum=1.5,
                    value=(
                        RUNTIME_CONFIG[
                            "repetition_penalty"
                        ]
                    ),
                    step=0.01,
                    label=(
                        "Repetition penalty"
                    ),
                )
            )

        gr.Markdown(
            "## Ответ"
        )

        answer_output = gr.Markdown(
            value=""
        )

        with gr.Accordion(
            "Диагностика",
            open=True,
        ):
            diagnostic_output = (
                gr.Markdown(
                    value=(
                        "### Диагностика\n\n"
                        "Ожидание запроса."
                    )
                )
            )

        inputs = [
            query,
            selected_mode,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        ]

        outputs = [
            answer_output,
            diagnostic_output,
        ]

        submit_button.click(
            fn=answer_query,
            inputs=inputs,
            outputs=outputs,
        )

        query.submit(
            fn=answer_query,
            inputs=inputs,
            outputs=outputs,
        )

        clear_button.click(
            fn=clear_interface,
            inputs=[],
            outputs=[
                query,
                answer_output,
                diagnostic_output,
            ],
        )

    return demo


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    demo = build_interface()

    print()
    print("=" * 80)
    print(
        "Starting NOVA web interface..."
    )
    print(
        "Open: http://127.0.0.1:7860"
    )
    print("=" * 80)
    print()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )