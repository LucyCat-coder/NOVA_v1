from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Embedder:
    """
    Sentence embedder для NOVA RAG.

    Поддерживает два режима pooling:

    legacy_mean
        Старый алгоритм FINAL NOVA.
        Нужен для совместимости с оригинальным
        knowledge_index.faiss.

    masked_mean
        Новый корректный mean pooling.
        Padding-токены не участвуют.
        Используется для knowledge_index_v2.

    ВАЖНО:
    Нельзя использовать masked_mean для старого
    индекса и legacy_mean для нового.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    VALID_POOLING_MODES = {
        "legacy_mean",
        "masked_mean",
    }

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        pooling_mode: str = "legacy_mean",
    ):
        if pooling_mode not in self.VALID_POOLING_MODES:
            raise ValueError(
                "Неизвестный pooling_mode: "
                f"{pooling_mode!r}. "
                f"Допустимо: "
                f"{sorted(self.VALID_POOLING_MODES)}"
            )

        self.model_name = model_name
        self.pooling_mode = pooling_mode

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        print(
            f"[Embedder] Loading tokenizer: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        print(
            f"[Embedder] Loading model: "
            f"{model_name}"
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.dim = int(
            self.model.config.hidden_size
        )

        print(
            "[Embedder] Ready: "
            f"dim={self.dim}, "
            f"device={self.device}, "
            f"pooling={self.pooling_mode}"
        )

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_mode == "legacy_mean":
            # Оригинальная FINAL NOVA.
            #
            # Padding также попадает в mean.
            # Оставляем только ради старого индекса.
            return (
                last_hidden_state
                .mean(dim=1)
            )

        if self.pooling_mode == "masked_mean":
            # Правильный mean pooling:
            # учитываем только реальные токены.
            mask = (
                attention_mask
                .unsqueeze(-1)
                .to(
                    dtype=last_hidden_state.dtype
                )
            )

            summed = (
                last_hidden_state
                * mask
            ).sum(
                dim=1
            )

            counts = (
                mask
                .sum(dim=1)
                .clamp_min(1e-8)
            )

            return (
                summed
                / counts
            )

        raise RuntimeError(
            "Недостижимый pooling_mode: "
            f"{self.pooling_mode}"
        )

    @torch.inference_mode()
    def _encode_raw(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, self.dim),
                dtype=np.float32,
            )

        all_embeddings = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                start:start + batch_size
            ]

            encoded = (
                self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            output = self.model(
                **encoded
            )

            embeddings = self._pool(
                last_hidden_state=(
                    output.last_hidden_state
                ),
                attention_mask=(
                    encoded["attention_mask"]
                ),
            )

            norms = (
                embeddings
                .norm(
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(1e-8)
            )

            embeddings = (
                embeddings
                / norms
            )

            all_embeddings.append(
                embeddings
                .float()
                .cpu()
                .numpy()
            )

        return (
            np.concatenate(
                all_embeddings,
                axis=0,
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        prepared = [
            self.QUERY_PREFIX
            + text
            for text in texts
        ]

        return self._encode_raw(
            prepared,
            batch_size=batch_size,
        )

    def encode_passages(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        prepared = [
            self.PASSAGE_PREFIX
            + text
            for text in texts
        ]

        return self._encode_raw(
            prepared,
            batch_size=batch_size,
        )