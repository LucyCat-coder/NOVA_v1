from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    FAISS IndexFlatIP + тексты и metadata.

    Существующий FINAL NOVA index хранится как:

        knowledge_index.faiss
        knowledge_index.jsonl
    """

    def __init__(
        self,
        dim: int,
    ):
        self.dim = int(dim)

        self.index = (
            faiss.IndexFlatIP(
                self.dim
            )
        )

        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def __len__(self) -> int:
        return len(
            self.texts
        )

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def ntotal(self) -> int:
        return int(
            self.index.ntotal
        )

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ):
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings должны иметь "
                "форму (N, dim)"
            )

        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"texts={len(texts)}, "
                f"embeddings={embeddings.shape[0]}"
            )

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"{embeddings.shape[1]} "
                f"!= {self.dim}"
            )

        if (
            metadata is not None
            and len(metadata) != len(texts)
        ):
            raise ValueError(
                "metadata и texts должны "
                "иметь одинаковую длину."
            )

        vectors = embeddings.astype(
            np.float32,
            copy=False,
        )

        self.index.add(
            vectors
        )

        self.texts.extend(
            texts
        )

        if metadata is None:
            self.metadata.extend(
                [{} for _ in texts]
            )
        else:
            self.metadata.extend(
                metadata
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[
        tuple[
            str,
            float,
            dict,
        ]
    ]:
        if self.is_empty:
            return []

        if query_embedding.ndim != 2:
            raise ValueError(
                "query_embedding должен иметь "
                "форму (N, dim)"
            )

        if query_embedding.shape[0] != 1:
            raise ValueError(
                "search сейчас принимает "
                "ровно один query vector."
            )

        if query_embedding.shape[1] != self.dim:
            raise ValueError(
                "Query dimension mismatch: "
                f"{query_embedding.shape[1]} "
                f"!= {self.dim}"
            )

        top_k = max(
            1,
            min(
                int(top_k),
                len(self.texts),
            ),
        )

        query_embedding = (
            query_embedding.astype(
                np.float32,
                copy=False,
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k,
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            index = int(index)

            if index < 0:
                continue

            results.append(
                (
                    self.texts[index],
                    float(score),
                    self.metadata[index],
                )
            )

        return results

    def clear(self):
        self.index.reset()
        self.texts = []
        self.metadata = []

    def save(
        self,
        path: str | Path,
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss_path = (
            path.with_suffix(
                ".faiss"
            )
        )

        jsonl_path = (
            path.with_suffix(
                ".jsonl"
            )
        )

        faiss.write_index(
            self.index,
            str(faiss_path),
        )

        with jsonl_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for text, metadata in zip(
                self.texts,
                self.metadata,
            ):
                record = {
                    "text": text,
                    "meta": metadata,
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "VectorStore":
        path = Path(path)

        faiss_path = (
            path.with_suffix(
                ".faiss"
            )
        )

        jsonl_path = (
            path.with_suffix(
                ".jsonl"
            )
        )

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index не найден: "
                f"{faiss_path}"
            )

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Metadata не найдены: "
                f"{jsonl_path}"
            )

        print(
            f"[VectorStore] Loading: "
            f"{faiss_path}"
        )

        index = faiss.read_index(
            str(faiss_path)
        )

        store = cls(
            dim=int(index.d)
        )

        store.index = index

        print(
            f"[VectorStore] FAISS: "
            f"dim={index.d}, "
            f"ntotal={index.ntotal}"
        )

        with jsonl_path.open(
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

                try:
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Некорректный JSON "
                        f"в строке {line_number}: "
                        f"{jsonl_path}"
                    ) from exc

                if "text" not in record:
                    raise RuntimeError(
                        "В metadata отсутствует "
                        f"'text', строка "
                        f"{line_number}"
                    )

                store.texts.append(
                    record["text"]
                )

                store.metadata.append(
                    record.get(
                        "meta",
                        {},
                    )
                )

        if (
            len(store.texts)
            != store.index.ntotal
        ):
            raise RuntimeError(
                "FAISS и JSONL содержат "
                "разное число записей: "
                f"faiss={store.index.ntotal}, "
                f"jsonl={len(store.texts)}"
            )

        print(
            "[VectorStore] Metadata: "
            f"{len(store.texts)} records"
        )

        return store