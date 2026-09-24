from __future__ import annotations

from pathlib import Path

from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    """
    Embedder + существующий FAISS index.

    Reranker пока намеренно не включаем
    в первый retrieval test.
    """

    def __init__(
        self,
        dim: int | None = None,
        embedder: Embedder | None = None,
        index_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.embedder = (
            embedder
            if embedder is not None
            else Embedder(
                device=device
            )
        )

        self.store = VectorStore(
            dim=(
                int(dim)
                if dim is not None
                else self.embedder.dim
            )
        )

        self._reranker = None

        if index_path is not None:
            self.load_index(
                index_path
            )

    def __len__(self) -> int:
        return len(
            self.store
        )

    @property
    def is_empty(self) -> bool:
        return self.store.is_empty

    def load_index(
        self,
        path: str | Path,
    ):
        self.store = (
            VectorStore.load(
                path
            )
        )

        if (
            self.store.dim
            != self.embedder.dim
        ):
            raise RuntimeError(
                "Размерность FAISS index "
                "не совпадает с embedder: "
                f"index={self.store.dim}, "
                f"embedder={self.embedder.dim}"
            )

    def save_index(
        self,
        path: str | Path,
    ):
        self.store.save(
            path
        )

    def index_documents(
        self,
        docs: list[str],
        metadata: list[dict] | None = None,
        batch_size: int = 64,
    ):
        if not docs:
            return

        embeddings = (
            self.embedder
            .encode_passages(
                docs,
                batch_size=batch_size,
            )
        )

        self.store.add(
            docs,
            embeddings,
            metadata,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
        use_reranker: bool = False,
        rerank_top_k: int | None = None,
    ) -> list[
        tuple[
            str,
            float,
            dict,
        ]
    ]:
        query = query.strip()

        if not query:
            return []

        if self.is_empty:
            return []

        faiss_k = (
            top_k * 3
            if use_reranker
            else top_k
        )

        query_embedding = (
            self.embedder
            .encode_queries(
                [query]
            )
        )

        results = (
            self.store.search(
                query_embedding,
                top_k=faiss_k,
            )
        )

        if min_score is not None:
            results = [
                item
                for item in results
                if item[1] >= min_score
            ]

        if (
            use_reranker
            and results
        ):
            if self._reranker is None:
                from .reranker import (
                    Reranker
                )

                self._reranker = (
                    Reranker(
                        device=str(
                            self.embedder.device
                        )
                    )
                )

            results = (
                self._reranker.rerank(
                    query,
                    results,
                    top_k=(
                        rerank_top_k
                        or top_k
                    ),
                )
            )

        return results[
            :top_k
        ]

    def retrieve_texts(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[str]:
        return [
            text
            for text, _, _
            in self.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                use_reranker=False,
            )
        ]