from __future__ import annotations

from pathlib import Path

from .embedder import Embedder
from .retriever import Retriever


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_CORE_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "core_index"
)

DEFAULT_TECHNICAL_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "technical_index"
)

DEFAULT_BROAD_INDEX = (
    PROJECT_ROOT
    / "rag"
    / "knowledge_index_v2"
)


class SafeRetriever:
    """
    Безопасный маршрутизатор RAG для NOVA.

    AUTO:
        Использует только маленький
        курируемый CORE.

        Если top-1 CORE score >= threshold,
        возвращается один CORE-документ.

        Иначе возвращается пустой список:
        NOVA отвечает без RAG.

    TECHNICAL:
        Явный ручной режим.
        Использует nova_technical_v1_1.

    BROAD:
        Явный ручной режим.
        Использует широкий knowledge_index_v2.

    OFF:
        RAG полностью отключён.

    ВАЖНО:
        Technical и broad никогда
        автоматически не подмешиваются
        в режиме auto.
    """

    VALID_MODES = {
        "auto",
        "core",
        "technical",
        "broad",
        "off",
    }

    def __init__(
        self,
        *,
        core_index_path: str | Path = DEFAULT_CORE_INDEX,
        technical_index_path: str | Path = DEFAULT_TECHNICAL_INDEX,
        broad_index_path: str | Path = DEFAULT_BROAD_INDEX,
        device: str = "cuda",
        model_name: str = "intfloat/multilingual-e5-small",
        core_threshold: float = 0.83,
    ):
        self.core_index_path = Path(
            core_index_path
        )

        self.technical_index_path = Path(
            technical_index_path
        )

        self.broad_index_path = Path(
            broad_index_path
        )

        self.device = device

        self.model_name = model_name

        self.core_threshold = float(
            core_threshold
        )

        if not (
            0.0
            <= self.core_threshold
            <= 1.0
        ):
            raise ValueError(
                "core_threshold должен быть "
                "между 0 и 1."
            )

        self._check_index(
            self.core_index_path,
            required=True,
        )

        self.embedder = Embedder(
            model_name=self.model_name,
            device=self.device,
            pooling_mode="masked_mean",
        )

        self.core = Retriever(
            index_path=self.core_index_path,
            embedder=self.embedder,
        )

        self._technical = None
        self._broad = None

    @staticmethod
    def _check_index(
        base_path: Path,
        *,
        required: bool,
    ) -> bool:
        faiss_path = Path(
            str(base_path)
            + ".faiss"
        )

        jsonl_path = Path(
            str(base_path)
            + ".jsonl"
        )

        exists = (
            faiss_path.exists()
            and jsonl_path.exists()
        )

        if (
            required
            and not exists
        ):
            raise FileNotFoundError(
                "RAG index отсутствует:\n"
                f"  {faiss_path}\n"
                f"  {jsonl_path}"
            )

        return exists

    @property
    def technical(
        self,
    ) -> Retriever:
        if self._technical is None:
            self._check_index(
                self.technical_index_path,
                required=True,
            )

            self._technical = Retriever(
                index_path=(
                    self.technical_index_path
                ),
                embedder=self.embedder,
            )

        return self._technical

    @property
    def broad(
        self,
    ) -> Retriever:
        if self._broad is None:
            self._check_index(
                self.broad_index_path,
                required=True,
            )

            self._broad = Retriever(
                index_path=(
                    self.broad_index_path
                ),
                embedder=self.embedder,
            )

        return self._broad

    def retrieve_with_info(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        min_score: float | None = None,
    ) -> dict:
        query = str(
            query
        ).strip()

        if not query:
            raise ValueError(
                "query не должен быть пустым."
            )

        mode = str(
            mode
        ).strip().lower()

        if mode not in self.VALID_MODES:
            raise ValueError(
                "Неизвестный RAG mode: "
                f"{mode!r}. "
                "Допустимые режимы: "
                + ", ".join(
                    sorted(
                        self.VALID_MODES
                    )
                )
            )

        if top_k <= 0:
            raise ValueError(
                "top_k должен быть > 0."
            )

        if mode == "off":
            return {
                "mode": mode,
                "route": "off",
                "accepted": False,
                "top1_score": None,
                "threshold": None,
                "results": [],
            }

        # ----------------------------------------------------
        # AUTO
        #
        # Только CORE.
        # Никакого автоматического fallback
        # в technical или broad.
        # ----------------------------------------------------

        if mode == "auto":
            core_results = (
                self.core.retrieve(
                    query=query,
                    top_k=max(
                        3,
                        top_k,
                    ),
                    min_score=None,
                    use_reranker=False,
                )
            )

            if not core_results:
                return {
                    "mode": mode,
                    "route": "none",
                    "accepted": False,
                    "top1_score": None,
                    "threshold": (
                        self.core_threshold
                    ),
                    "results": [],
                }

            top1_score = float(
                core_results[0][1]
            )

            accepted = (
                top1_score
                >= self.core_threshold
            )

            if accepted:
                # В AUTO подаём модели
                # только один проверенный
                # CORE answer.
                selected = [
                    core_results[0]
                ]

                route = "core"

            else:
                selected = []
                route = "none"

            return {
                "mode": mode,
                "route": route,
                "accepted": accepted,
                "top1_score": top1_score,
                "threshold": (
                    self.core_threshold
                ),
                "results": selected,
            }

        # ----------------------------------------------------
        # CORE MANUAL
        # ----------------------------------------------------

        if mode == "core":
            results = self.core.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                use_reranker=False,
            )

            return {
                "mode": mode,
                "route": "core",
                "accepted": bool(
                    results
                ),
                "top1_score": (
                    float(
                        results[0][1]
                    )
                    if results
                    else None
                ),
                "threshold": min_score,
                "results": results,
            }

        # ----------------------------------------------------
        # TECHNICAL MANUAL
        # ----------------------------------------------------

        if mode == "technical":
            results = (
                self.technical.retrieve(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    use_reranker=False,
                )
            )

            return {
                "mode": mode,
                "route": "technical",
                "accepted": bool(
                    results
                ),
                "top1_score": (
                    float(
                        results[0][1]
                    )
                    if results
                    else None
                ),
                "threshold": min_score,
                "results": results,
            }

        # ----------------------------------------------------
        # BROAD MANUAL
        # ----------------------------------------------------

        results = self.broad.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            use_reranker=False,
        )

        return {
            "mode": mode,
            "route": "broad",
            "accepted": bool(
                results
            ),
            "top1_score": (
                float(
                    results[0][1]
                )
                if results
                else None
            ),
            "threshold": min_score,
            "results": results,
        }

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        min_score: float | None = None,
    ):
        result = self.retrieve_with_info(
            query=query,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
        )

        return result[
            "results"
        ]