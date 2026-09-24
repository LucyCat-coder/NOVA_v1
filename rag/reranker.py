from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class Reranker:
    """
    Cross-encoder reranker для RAG.

    Получает query + несколько passage-кандидатов
    от FAISS и пересортировывает их по смысловой
    релевантности.

    ВАЖНО:
    cross_score здесь НЕ является cosine similarity
    и его нельзя напрямую сравнивать с FAISS score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(
            f"[Reranker] Loading tokenizer: "
            f"{self.model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        print(
            f"[Reranker] Loading model: "
            f"{self.model_name}"
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "[Reranker] Ready: "
            f"device={self.device}"
        )

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        candidates: list[
            tuple[
                str,
                float,
                dict,
            ]
        ],
        top_k: int = 1,
        batch_size: int = 16,
    ) -> list[
        tuple[
            str,
            float,
            dict,
        ]
    ]:
        """
        candidates:
            [
                (
                    passage_text,
                    faiss_score,
                    metadata
                ),
                ...
            ]

        return:
            [
                (
                    passage_text,
                    cross_score,
                    metadata
                ),
                ...
            ]

        Сортировка идёт по cross_score.
        """

        query = query.strip()

        if not query:
            return []

        if not candidates:
            return []

        top_k = max(
            1,
            min(
                int(top_k),
                len(candidates),
            ),
        )

        scored = []

        for start in range(
            0,
            len(candidates),
            batch_size,
        ):
            batch = candidates[
                start:start + batch_size
            ]

            passages = [
                item[0]
                for item in batch
            ]

            queries = [
                query
                for _ in passages
            ]

            encoded = (
                self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
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

            logits = (
                output.logits
                .float()
                .cpu()
            )

            if (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):
                scores = (
                    logits[:, 0]
                    .tolist()
                )

            elif (
                logits.ndim == 2
                and logits.shape[1] == 2
            ):
                # На случай binary classifier
                # с двумя logits.
                scores = (
                    (
                        logits[:, 1]
                        - logits[:, 0]
                    )
                    .tolist()
                )

            else:
                raise RuntimeError(
                    "Неожиданная форма logits "
                    "reranker: "
                    f"{tuple(logits.shape)}"
                )

            for candidate, score in zip(
                batch,
                scores,
            ):
                text, _, metadata = (
                    candidate
                )

                scored.append(
                    (
                        text,
                        float(score),
                        metadata,
                    )
                )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored[
            :top_k
        ]