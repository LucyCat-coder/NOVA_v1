from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_CONFIG = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "block_size": 512,
}


RUNTIME_CONFIG = {
    "device": "cuda",

    "checkpoint": (
        PROJECT_ROOT
        / "out"
        / "final"
        / "best_final.pt"
    ),

    # ========================================================
    # SAFE RAG
    # ========================================================

    "rag_core_index": (
        PROJECT_ROOT
        / "rag"
        / "core_index"
    ),

    "rag_technical_index": (
        PROJECT_ROOT
        / "rag"
        / "technical_index"
    ),

    "rag_broad_index": (
        PROJECT_ROOT
        / "rag"
        / "knowledge_index_v2"
    ),

    # По CORE gate tests:
    #
    # positive min = 0.850382
    # negative max = 0.808870
    #
    # 0.83 находится между ними.
    "rag_core_threshold": 0.83,

    # Старый alias.
    #
    # Оставлен только для совместимости
    # со старым app.py.
    "rag_index": (
        PROJECT_ROOT
        / "rag"
        / "knowledge_index_v2"
    ),

    # ========================================================
    # GENERATION
    # ========================================================

    "max_new_tokens": 200,
    "temperature": 0.5,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.15,

    # Пока False.
    #
    # До изменения app.py мы проверяем
    # новую схему отдельным тестом.
    "rag_enabled": False,

    # AUTO:
    #
    # CORE >= threshold
    #     -> direct trusted answer
    #
    # иначе
    #     -> FINAL NOVA without RAG
    "rag_source": "auto",

    # auto:
    #
    # CORE / technical
    #     -> direct retrieval answer
    #
    # broad / no RAG
    #     -> model generation
    "rag_answer_mode": "auto",

    # Используется только в generated
    # RAG path, главным образом broad.
    "rag_mode": "inline",

    "top_k_docs": 1,

    # AUTO CORE использует
    # rag_core_threshold.
    #
    # Здесь min_score нужен для
    # ручных source modes.
    "min_score": 0.80,

    "use_reranker": False,
}


CHECKPOINTS = {
    "final": (
        PROJECT_ROOT
        / "out"
        / "final"
        / "best_final.pt"
    ),

    "nova_v1": (
        PROJECT_ROOT
        / "out"
        / "archive"
        / "nova_v1"
        / "checkpoint.pt"
    ),

    "technical_v1_1": (
        PROJECT_ROOT
        / "out"
        / "archive"
        / "technical_v1_1"
        / "best_checkpoint.pt"
    ),
}